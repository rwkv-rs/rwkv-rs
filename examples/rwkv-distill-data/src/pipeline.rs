use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    fs::{self, File, OpenOptions},
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, ensure};
use futures::{StreamExt, stream};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sonic_rs::{JsonValueTrait, Value};

use crate::config::{Config, GeneratorConfig, ModelConfig, RunConfig, load_config, validate_config};

#[derive(Clone)]
struct SourceSample {
    sample_id: String,
    source_user: String,
}

#[derive(Clone)]
struct GenerateJob {
    sample: SourceSample,
    missing_indices: Vec<usize>,
}

#[derive(Clone)]
struct PendingTask {
    task_id: String,
    rewritten_user: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TaskStateRow {
    task_id: String,
    status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    rewritten_user: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    answer_model: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TrainRow {
    task_id: String,
    answer_model: String,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunStatus {
    Generated,
    Done,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Clone, Copy, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Clone)]
struct OpenAiClient {
    http: Client,
    endpoint: String,
    api_key: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ChatResult {
    content: String,
    reasoning: Option<String>,
}

#[derive(Debug)]
struct RequestError {
    transient: bool,
    message: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct GeneratedQuestions {
    questions: Vec<GeneratedQuestion>,
}

#[derive(Deserialize)]
struct GeneratedQuestion {
    user: String,
}

struct ResumePlan {
    generator_jobs: Vec<GenerateJob>,
    pending_tasks: Vec<PendingTask>,
    resumed_generated: usize,
    resumed_done: usize,
    recovered_done_without_train: usize,
}

struct AnswerCompletion {
    state_row: TaskStateRow,
    train_row: TrainRow,
}

pub async fn synthesize(path: &Path, limit: Option<usize>) -> Result<()> {
    let mut cfg = load_config(path)?;
    if let Some(limit) = limit {
        cfg.input.limit = Some(limit);
    }
    validate_config(&cfg)?;

    let samples = load_samples(&cfg)?;
    if samples.is_empty() {
        println!("no samples");
        return Ok(());
    }

    prepare_output(&cfg)?;

    let state_rows = if cfg.run.resume {
        load_state_rows(&cfg.output.state_jsonl_path)?
    } else {
        HashMap::new()
    };
    let train_rows = if cfg.run.resume {
        load_train_rows(&cfg.output.train_jsonl_path)?
    } else {
        HashMap::new()
    };

    if !state_rows.is_empty() {
        let (generated, done) = summarize_state_rows(&state_rows)?;
        eprintln!(
            "loaded {} task states from {} (generated={}, done={})",
            state_rows.len(),
            cfg.output.state_jsonl_path.display(),
            generated,
            done
        );
    }
    if !train_rows.is_empty() {
        eprintln!(
            "loaded {} finished training rows from {}",
            train_rows.len(),
            cfg.output.train_jsonl_path.display()
        );
    }

    let plan = build_resume_plan(
        &samples,
        cfg.generator.variant_count,
        &state_rows,
        &train_rows,
    )?;
    if plan.recovered_done_without_train > 0 {
        eprintln!(
            "recovered {} tasks whose state said done but train row was missing; they will be answered again",
            plan.recovered_done_without_train
        );
    }

    let mut pending_tasks = plan.pending_tasks;
    let mut generated_now = 0usize;
    let mut skipped_generate = 0usize;

    if !plan.generator_jobs.is_empty() {
        let client = OpenAiClient::new(&cfg.generator.model, &cfg.run)?;
        let generator = cfg.generator.clone();
        let pb = progress_bar(plan.generator_jobs.len(), "generate");
        let concurrency =
            concurrency_limit(cfg.concurrency.generate_requests, plan.generator_jobs.len());
        let mut stream = stream::iter(plan.generator_jobs)
            .map(|job| {
                let client = client.clone();
                let generator = generator.clone();
                async move { generate_tasks(client, generator, job).await }
            })
            .buffer_unordered(concurrency);

        while let Some(result) = stream.next().await {
            match result {
                Ok(tasks) => {
                    let generated_rows = tasks.iter().map(task_state_generated).collect::<Vec<_>>();
                    append_jsonl(&cfg.output.state_jsonl_path, &generated_rows)?;
                    generated_now += tasks.len();
                    pending_tasks.extend(tasks);
                }
                Err(err) => {
                    skipped_generate += 1;
                    eprintln!("skipped generate job: {err:#}");
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("generate done");
    }

    if pending_tasks.is_empty() {
        println!(
            "samples={} total_tasks={} resumed_done={} resumed_generated={} generated_now=0 answered_now=0 skipped_generate={} skipped_answer=0",
            samples.len(),
            samples.len() * cfg.generator.variant_count,
            plan.resumed_done,
            plan.resumed_generated,
            skipped_generate,
        );
        return Ok(());
    }

    let answer_clients = cfg
        .answer_models
        .iter()
        .map(|model| OpenAiClient::new(model, &cfg.run))
        .collect::<Result<Vec<_>>>()?;
    let answer_models = cfg.answer_models.clone();
    let answer_total = pending_tasks.len();
    let answer_pb = progress_bar(answer_total, "answer");
    let concurrency = concurrency_limit(cfg.concurrency.answer_requests, answer_total);
    let mut answered_now = 0usize;
    let mut skipped_answer = 0usize;

    let mut stream = stream::iter(pending_tasks)
        .map(|task| {
            let idx = pick_model_index(&task.task_id, answer_models.len());
            let client = answer_clients[idx].clone();
            let model = answer_models[idx].clone();
            async move { answer_task(client, model, task).await }
        })
        .buffer_unordered(concurrency);

    while let Some(result) = stream.next().await {
        match result {
            Ok(done) => {
                append_jsonl(
                    &cfg.output.train_jsonl_path,
                    std::slice::from_ref(&done.train_row),
                )?;
                append_jsonl(
                    &cfg.output.state_jsonl_path,
                    std::slice::from_ref(&done.state_row),
                )?;
                answered_now += 1;
            }
            Err(err) => {
                skipped_answer += 1;
                eprintln!("skipped answer task: {err:#}");
            }
        }
        answer_pb.inc(1);
    }
    answer_pb.finish_with_message("answer done");

    println!(
        "samples={} total_tasks={} resumed_done={} resumed_generated={} generated_now={} answered_now={} skipped_generate={} skipped_answer={}",
        samples.len(),
        samples.len() * cfg.generator.variant_count,
        plan.resumed_done,
        plan.resumed_generated,
        generated_now,
        answered_now,
        skipped_generate,
        skipped_answer
    );
    Ok(())
}

async fn generate_tasks(
    client: OpenAiClient,
    generator: GeneratorConfig,
    job: GenerateJob,
) -> Result<Vec<PendingTask>> {
    let question_count = job.missing_indices.len();
    ensure!(question_count > 0, "generate job is empty");

    let prompt = build_generation_prompt(&job.sample, question_count)?;
    let result = client
        .chat(&generator.model, &prompt, true)
        .await
        .with_context(|| format!("generator failed for sample {}", job.sample.sample_id))?;
    let users = parse_generated_users(&result.content, question_count)?;

    job.missing_indices
        .into_iter()
        .zip(users.into_iter())
        .map(|(index, user)| {
            Ok(PendingTask {
                task_id: task_id(&job.sample.sample_id, index),
                rewritten_user: user,
            })
        })
        .collect()
}

async fn answer_task(
    client: OpenAiClient,
    model: ModelConfig,
    task: PendingTask,
) -> Result<AnswerCompletion> {
    let result = client
        .chat(&model, &task.rewritten_user, false)
        .await
        .with_context(|| format!("answer failed for task {}", task.task_id))?;
    let assistant = merge_assistant(result.reasoning, result.content);
    let text = rwkv_text(&task.rewritten_user, &assistant);

    Ok(AnswerCompletion {
        state_row: TaskStateRow {
            task_id: task.task_id.clone(),
            status: RunStatus::Done.as_str().to_owned(),
            rewritten_user: task.rewritten_user,
            answer_model: model.model_name.clone(),
        },
        train_row: TrainRow {
            task_id: task.task_id,
            answer_model: model.model_name,
            text,
        },
    })
}

impl OpenAiClient {
    fn new(model: &ModelConfig, run: &RunConfig) -> Result<Self> {
        let mut builder =
            Client::builder().timeout(Duration::from_secs_f64(run.request_timeout_seconds));
        if run.disable_env_proxy {
            builder = builder.no_proxy();
        }
        if run.force_http1 {
            builder = builder.http1_only();
        }
        Ok(Self {
            http: builder.build()?,
            endpoint: model.endpoint.clone(),
            api_key: model.api_key.clone(),
        })
    }

    async fn chat(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> Result<ChatResult> {
        let max_attempts = 3usize;
        for attempt in 0..max_attempts {
            match self.try_chat(model, prompt, json_output).await {
                Ok(result) => return Ok(result),
                Err(err) if err.transient && attempt + 1 < max_attempts => {
                    eprintln!(
                        "transient chat error for model {} (attempt {}/{}): {}",
                        model.model_name,
                        attempt + 1,
                        max_attempts,
                        err
                    );
                    tokio::time::sleep(Duration::from_secs((attempt + 1) as u64 * 2)).await;
                }
                Err(err) => return Err(err.into()),
            }
        }
        unreachable!("chat retry loop should either return or fail")
    }

    async fn try_chat(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<ChatResult, RequestError> {
        let body = self.send_chat_request(model, prompt, json_output).await?;
        parse_chat_result(&body).map_err(RequestError::parse)
    }

    async fn send_chat_request(
        &self,
        model: &ModelConfig,
        prompt: &str,
        json_output: bool,
    ) -> std::result::Result<String, RequestError> {
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&chat_request(model, prompt, json_output))
            .send()
            .await
            .map_err(RequestError::http)?;
        let status = response.status();
        let body = response.text().await.map_err(RequestError::http)?;
        if !status.is_success() {
            return Err(RequestError::api(status, body));
        }
        Ok(body)
    }
}

impl RequestError {
    fn http(err: reqwest::Error) -> Self {
        Self {
            transient: err.is_timeout() || err.is_connect() || err.is_body(),
            message: err.to_string(),
        }
    }

    fn api(status: StatusCode, body: String) -> Self {
        Self {
            transient: status.is_server_error() || status.as_u16() == 429,
            message: format!(
                "chat completion failed with status {}: {}",
                status,
                preview_text(&body, 400)
            ),
        }
    }

    fn parse(err: anyhow::Error) -> Self {
        Self {
            transient: false,
            message: err.to_string(),
        }
    }
}

impl Display for RequestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

fn chat_request<'a>(model: &'a ModelConfig, prompt: &'a str, json_output: bool) -> ChatRequest<'a> {
    ChatRequest {
        model: &model.model_name,
        messages: [ChatMessage {
            role: "user",
            content: prompt,
        }],
        max_completion_tokens: model.max_completion_tokens,
        reasoning_effort: model.reasoning_effort.as_deref(),
        response_format: json_output.then(json_object_response_format),
    }
}

fn json_object_response_format() -> ResponseFormat {
    ResponseFormat {
        kind: "json_object",
    }
}

fn load_samples(cfg: &Config) -> Result<Vec<SourceSample>> {
    let mut out = Vec::new();
    let mut invalid_count = 0usize;
    let mut invalid_examples = Vec::new();

    for (index, value) in load_input_rows_window(
        &cfg.input.dataset_path,
        cfg.input.start_index,
        cfg.input.limit,
    )? {
        match normalize_sample(index, value) {
            Ok(sample) => out.push(sample),
            Err(err) => {
                invalid_count += 1;
                if invalid_examples.len() < 5 {
                    invalid_examples.push(format!("sample_index={index}: {err}"));
                }
            }
        }
    }

    if invalid_count > 0 {
        let preview = if invalid_examples.is_empty() {
            String::new()
        } else {
            format!(" examples: {}", invalid_examples.join(" | "))
        };
        eprintln!(
            "skipped {invalid_count} invalid normalized samples from {}{preview}",
            cfg.input.dataset_path.display()
        );
    }

    Ok(out)
}

fn normalize_sample(_index: usize, value: Value) -> Result<SourceSample> {
    let task_id = required_top_level_text(&value, "task_id")?;
    let sample_index = required_top_level_text(&value, "sample_index")?;
    let completions_id = required_top_level_text(&value, "completions_id")?;
    let context = required_top_level_text(&value, "context")?;
    let source_user = parse_single_turn_context(&context).with_context(|| {
        format!("invalid single-turn context for sample {task_id}_{sample_index}_{completions_id}")
    })?;

    Ok(SourceSample {
        sample_id: format!("{task_id}_{sample_index}_{completions_id}"),
        source_user,
    })
}

fn parse_single_turn_context(context: &str) -> Result<String> {
    let trimmed = context.trim();
    ensure!(!trimmed.is_empty(), "context is empty");

    let user_headers = header_positions(trimmed, "User:");
    let assistant_headers = header_positions(trimmed, "Assistant:");
    ensure!(
        user_headers.len() == 1,
        "context must contain exactly one line-start User: header, got {} in {:?}",
        user_headers.len(),
        preview_text(trimmed, 120)
    );
    ensure!(
        assistant_headers.len() == 1,
        "context must contain exactly one line-start Assistant: header, got {} in {:?}",
        assistant_headers.len(),
        preview_text(trimmed, 120)
    );

    let user_start = user_headers[0];
    let assistant_start = assistant_headers[0];
    ensure!(user_start == 0, "context must start with User:");
    ensure!(
        assistant_start > user_start,
        "Assistant: header must appear after User: header"
    );

    let user = trimmed["User:".len()..assistant_start].trim();
    ensure!(!user.is_empty(), "context user content is empty");
    Ok(user.to_owned())
}

fn header_positions(text: &str, header: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut offset = 0usize;

    for segment in text.split_inclusive('\n') {
        let line = segment.trim_end_matches(&['\r', '\n'][..]);
        if line.starts_with(header) {
            positions.push(offset);
        }
        offset += segment.len();
    }

    if !text.ends_with('\n') {
        let line_start = text.rfind('\n').map(|idx| idx + 1).unwrap_or(0);
        let last_line = &text[line_start..];
        if last_line.starts_with(header) && positions.last().copied() != Some(line_start) {
            positions.push(line_start);
        }
    }

    positions
}

fn build_resume_plan(
    samples: &[SourceSample],
    variant_count: usize,
    state_rows: &HashMap<String, TaskStateRow>,
    train_rows: &HashMap<String, TrainRow>,
) -> Result<ResumePlan> {
    let mut generator_jobs = Vec::new();
    let mut pending_tasks = Vec::new();
    let mut resumed_generated = 0usize;
    let mut resumed_done = 0usize;
    let mut recovered_done_without_train = 0usize;

    for sample in samples {
        let mut missing_indices = Vec::new();

        for index in 0..variant_count {
            let id = task_id(&sample.sample_id, index);

            if train_rows.contains_key(&id) {
                resumed_done += 1;
                continue;
            }

            match state_rows.get(&id) {
                Some(row) => match parse_row_status(&row.status, &row.task_id)? {
                    RunStatus::Done => {
                        pending_tasks.push(pending_task_from_state_row(row)?);
                        recovered_done_without_train += 1;
                    }
                    RunStatus::Generated => {
                        pending_tasks.push(pending_task_from_state_row(row)?);
                        resumed_generated += 1;
                    }
                },
                None => missing_indices.push(index),
            }
        }

        if !missing_indices.is_empty() {
            generator_jobs.push(GenerateJob {
                sample: sample.clone(),
                missing_indices,
            });
        }
    }

    Ok(ResumePlan {
        generator_jobs,
        pending_tasks,
        resumed_generated,
        resumed_done,
        recovered_done_without_train,
    })
}

fn build_generation_prompt(sample: &SourceSample, question_count: usize) -> Result<String> {
    let payload = sonic_rs::json!({
        "sample_id": sample.sample_id.clone(),
        "user": sample.source_user.clone(),
    });
    let pretty = sonic_rs::to_string_pretty(&payload)?;

    Ok(format!(
        "You are rewriting user questions for RWKV training.\n\
Return only JSON.\n\
The response must be a single JSON object with exactly one key named \"questions\".\n\
The value of \"questions\" must be an array of exactly {question_count} objects.\n\
Each object must have exactly one key named \"user\".\n\
Each \"user\" value must be a standalone rewritten user question.\n\
Do not answer the question.\n\
Do not include markdown, code fences, commentary, or extra keys.\n\n\
Source user:\n{pretty}"
    ))
}

fn parse_generated_users(text: &str, expected: usize) -> Result<Vec<String>> {
    let payload: GeneratedQuestions =
        sonic_rs::from_str(text).context("generator did not return valid JSON")?;
    collect_users(&payload.questions, expected)
}

fn collect_users(items: &[GeneratedQuestion], expected: usize) -> Result<Vec<String>> {
    ensure!(
        items.len() == expected,
        "expected exactly {expected} generated questions, got {}",
        items.len()
    );
    items.iter().map(generated_user_text).collect()
}

fn generated_user_text(value: &GeneratedQuestion) -> Result<String> {
    let user = value.user.trim().to_owned();
    ensure!(!user.is_empty(), "generated user prompt is empty");
    Ok(user)
}

fn parse_chat_result(body: &str) -> Result<ChatResult> {
    let parsed: ChatResponse = sonic_rs::from_str(body)?;
    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("chat response is missing choices[0]"))?;
    let content = choice.message.content.unwrap_or_default().trim().to_owned();
    let reasoning = choice
        .message
        .reasoning_content
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty());

    ensure!(
        !content.is_empty() || reasoning.is_some(),
        "chat response is empty"
    );

    Ok(ChatResult { content, reasoning })
}

fn merge_assistant(reasoning: Option<String>, content: String) -> String {
    match (reasoning.map(|text| text.trim().to_owned()), content.trim()) {
        (Some(reasoning), "") => format!("<think>\n{reasoning}\n</think>"),
        (Some(reasoning), content) => format!("<think>\n{reasoning}\n</think>\n\n{content}"),
        (None, content) => content.to_owned(),
    }
}

fn rwkv_text(user: &str, assistant: &str) -> String {
    format!("User: {}\nAssistant: {}", user.trim(), assistant.trim())
}

fn task_state_generated(task: &PendingTask) -> TaskStateRow {
    TaskStateRow {
        task_id: task.task_id.clone(),
        status: RunStatus::Generated.as_str().to_owned(),
        rewritten_user: task.rewritten_user.clone(),
        answer_model: String::new(),
    }
}

fn task_id(sample_id: &str, variant_index: usize) -> String {
    format!("{sample_id}_q{variant_index:03}")
}

fn pick_model_index(task_id: &str, count: usize) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    task_id.hash(&mut hasher);
    (hasher.finish() as usize) % count
}

fn load_input_rows_window(
    path: &Path,
    start_index: usize,
    limit: Option<usize>,
) -> Result<Vec<(usize, Value)>> {
    let take_limit = limit.unwrap_or(usize::MAX);
    if take_limit == 0 {
        return Ok(Vec::new());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut logical_index = 0usize;
    let mut skipped = 0usize;
    let mut examples = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        if out.len() >= take_limit {
            break;
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let value = match sonic_rs::from_str::<Value>(&line) {
            Ok(value) => value,
            Err(err) => {
                skipped += 1;
                if examples.len() < 5 {
                    examples.push(format!("line {}: {err}", line_no + 1));
                }
                continue;
            }
        };

        if logical_index >= start_index {
            out.push((logical_index, value));
        }
        logical_index += 1;
    }

    if skipped > 0 {
        let preview = if examples.is_empty() {
            String::new()
        } else {
            format!(" examples: {}", examples.join(" | "))
        };
        eprintln!(
            "skipped {skipped} invalid input JSONL lines in {}{preview}",
            path.display()
        );
    }

    Ok(out)
}

fn read_jsonl<T: DeserializeOwned>(path: &Path, label: &str) -> Result<Vec<T>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = sonic_rs::from_str::<T>(&line).with_context(|| {
            format!(
                "invalid {label} JSONL line {} in {}",
                line_no + 1,
                path.display()
            )
        })?;
        out.push(parsed);
    }

    Ok(out)
}

fn read_jsonl_if_exists<T: DeserializeOwned>(path: &Path, label: &str) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_jsonl(path, label)
}

fn append_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);

    for row in rows {
        let json = sonic_rs::to_string(row)?;
        writer.write_all(json.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(())
}

fn prepare_output(cfg: &Config) -> Result<()> {
    if let Some(parent) = cfg.output.state_jsonl_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = cfg.output.train_jsonl_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if !cfg.run.resume {
        File::create(&cfg.output.state_jsonl_path)?;
        File::create(&cfg.output.train_jsonl_path)?;
    }
    Ok(())
}

fn load_state_rows(path: &Path) -> Result<HashMap<String, TaskStateRow>> {
    Ok(read_jsonl_if_exists::<TaskStateRow>(path, "state")?
        .into_iter()
        .map(|row| (row.task_id.clone(), row))
        .collect())
}

fn load_train_rows(path: &Path) -> Result<HashMap<String, TrainRow>> {
    Ok(read_jsonl_if_exists::<TrainRow>(path, "train")?
        .into_iter()
        .map(|row| (row.task_id.clone(), row))
        .collect())
}

fn summarize_state_rows(rows: &HashMap<String, TaskStateRow>) -> Result<(usize, usize)> {
    rows.values()
        .try_fold((0usize, 0usize), |(generated, done), row| {
            Ok(match parse_row_status(&row.status, &row.task_id)? {
                RunStatus::Generated => (generated + 1, done),
                RunStatus::Done => (generated, done + 1),
            })
        })
}

fn pending_task_from_state_row(row: &TaskStateRow) -> Result<PendingTask> {
    let user = row.rewritten_user.trim();
    ensure!(
        !user.is_empty(),
        "resume row is missing rewritten_user for task {}",
        row.task_id
    );
    Ok(PendingTask {
        task_id: row.task_id.clone(),
        rewritten_user: user.to_owned(),
    })
}

fn parse_row_status(status: &str, task_id: &str) -> Result<RunStatus> {
    match status.trim() {
        "generated" => Ok(RunStatus::Generated),
        "done" => Ok(RunStatus::Done),
        _ => Err(anyhow!(
            "unsupported output status {:?} for task {}",
            status,
            task_id
        )),
    }
}

fn required_top_level_text(value: &Value, key: &str) -> Result<String> {
    top_level_text(value, key).ok_or_else(|| anyhow!("missing {key}"))
}

fn top_level_text(value: &Value, key: &str) -> Option<String> {
    scalar_text(value.get(key)?)
}

fn scalar_text(value: &Value) -> Option<String> {
    let text = if value.is_null() {
        return None;
    } else if let Some(text) = value.as_str() {
        text.trim().to_owned()
    } else if let Some(number) = value.as_number() {
        number.to_string()
    } else if let Some(boolean) = value.as_bool() {
        boolean.to_string()
    } else {
        return None;
    };

    (!text.is_empty()).then_some(text)
}

fn preview_text(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    let char_count = trimmed.chars().count();
    if char_count <= limit {
        trimmed.to_owned()
    } else {
        format!("{}...", trimmed.chars().take(limit).collect::<String>())
    }
}

fn progress_bar(total: usize, label: &str) -> ProgressBar {
    if total == 0 {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:>10} [{bar:40.cyan/blue}] {pos}/{len} {percent:>3}% {elapsed_precise}<{eta_precise}",
        )
        .expect("valid progress bar template"),
    );
    pb.set_prefix(label.to_owned());
    pb
}

fn concurrency_limit(configured: usize, total: usize) -> usize {
    configured.max(1).min(total.max(1))
}

impl RunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generated => "generated",
            Self::Done => "done",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GeneratedQuestion, build_resume_plan, merge_assistant, parse_generated_users,
        parse_single_turn_context,
    };
    use std::collections::HashMap;

    use super::{SourceSample, TaskStateRow, TrainRow};

    #[test]
    fn parse_single_turn_context_extracts_user() {
        let user = parse_single_turn_context("User: hello\nAssistant: world").unwrap();
        assert_eq!(user, "hello");
    }

    #[test]
    fn merge_assistant_wraps_reasoning() {
        let merged = merge_assistant(Some("step by step".to_string()), "final".to_string());
        assert_eq!(merged, "<think>\nstep by step\n</think>\n\nfinal");
    }

    #[test]
    fn parse_generated_users_checks_count() {
        let users =
            parse_generated_users(r#"{"questions":[{"user":"q1"},{"user":"q2"}]}"#, 2).unwrap();
        assert_eq!(users, vec!["q1".to_string(), "q2".to_string()]);
    }

    #[test]
    fn resume_plan_prefers_train_rows_as_done() {
        let samples = vec![SourceSample {
            sample_id: "task_0_1".to_string(),
            source_user: "hello".to_string(),
        }];
        let mut state_rows = HashMap::new();
        state_rows.insert(
            "task_0_1_q000".to_string(),
            TaskStateRow {
                task_id: "task_0_1_q000".to_string(),
                status: "generated".to_string(),
                rewritten_user: "hello rewritten".to_string(),
                answer_model: String::new(),
            },
        );
        let mut train_rows = HashMap::new();
        train_rows.insert(
            "task_0_1_q000".to_string(),
            TrainRow {
                task_id: "task_0_1_q000".to_string(),
                answer_model: "answer-a".to_string(),
                text: "User: hello\nAssistant: world".to_string(),
            },
        );

        let plan = build_resume_plan(&samples, 1, &state_rows, &train_rows).unwrap();
        assert_eq!(plan.resumed_done, 1);
        assert!(plan.pending_tasks.is_empty());
        assert!(plan.generator_jobs.is_empty());
    }

    #[test]
    fn generated_question_struct_trims_later_in_validation() {
        let item = GeneratedQuestion {
            user: " hello ".to_string(),
        };
        assert_eq!(item.user, " hello ");
    }
}
