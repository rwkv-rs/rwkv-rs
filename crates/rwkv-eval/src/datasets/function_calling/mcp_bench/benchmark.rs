use super::judge::{
    collapse_evaluation_to_pass, evaluate_with_official_evaluator, resolve_judge_config,
    summarize_evaluation,
};
use super::prompt::{
    MCP_BENCH_COT_MAX_TOKENS, MCP_BENCH_DECISION_MAX_TOKENS, MCP_BENCH_FINAL_MAX_TOKENS,
    MCP_BENCH_MAX_ROUNDS, append_round_summary, build_context_summary, build_final_answer_prompt,
    build_planning_context, build_planning_decision_prompt, normalize_planned_tool_call,
    parse_planning_decision, render_trace,
};
use super::runtime::{
    MCP_BENCH_EXPECTED_LEN, MCP_BENCH_GIT_COMMIT, MCP_BENCH_RETRY_DELAY_SECS, MCP_BENCH_TASK_FILES,
    MCP_BENCH_TASK_RETRY_MAX, MCP_BENCH_TASK_TIMEOUT_SECS, McpBenchSession, check_runtime_checkout,
    ensure_runtime_checkout, mcp_bench_root, run_preflight, runtime_root as mcp_bench_runtime_root,
    sync_task_assets, task_assets_root as mcp_bench_task_assets_root,
};
use super::types::{
    McpBenchExecutionResult, McpBenchItem, McpBenchJudgeRequest, McpBenchPreflightSummary,
    McpBenchStep, PlannedToolCall, RawTaskFile,
};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use crate::inferers::generate_text_completion;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use sonic_rs::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};

#[distributed_slice(ALL_BENCHMARKS)]
static MCP_BENCH_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mcp_bench"),
    field: Field::FunctionCalling,
    display_name: "mcp_bench",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 0.3,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(McpBench::new(dataset_root)),
};

pub struct McpBench {
    dataset_root: PathBuf,
    test: Vec<McpBenchItem>,
    preflight: Mutex<Option<Result<McpBenchPreflightSummary, String>>>,
}

impl McpBench {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
            preflight: Mutex::new(None),
        }
    }

    fn task_assets_root(&self) -> PathBuf {
        mcp_bench_task_assets_root(&self.dataset_root)
    }

    fn runtime_root(&self) -> PathBuf {
        mcp_bench_runtime_root(&self.dataset_root)
    }

    fn load_items(&self) -> Result<Vec<McpBenchItem>, String> {
        let tasks_root = self.task_assets_root();
        if !tasks_root.is_dir() {
            return Err(format!(
                "missing MCP-Bench task assets under {}",
                tasks_root.display()
            ));
        }

        let mut items = Vec::new();
        for task_file in MCP_BENCH_TASK_FILES {
            let path = tasks_root.join(task_file);
            if !path.is_file() {
                return Err(format!("missing MCP-Bench task file: {}", path.display()));
            }

            let text = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            let raw = sonic_rs::from_str::<RawTaskFile>(&text)
                .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;

            for group in raw.server_tasks {
                for task in group.tasks {
                    items.push(McpBenchItem {
                        task_file: task_file.to_string(),
                        server_name: group.server_name.clone(),
                        combination_name: group.combination_name.clone(),
                        combination_type: group.combination_type.clone(),
                        servers: group.servers.clone(),
                        task,
                    });
                }
            }
        }
        Ok(items)
    }

    async fn ensure_preflight_once(&self) -> Result<McpBenchPreflightSummary, String> {
        let mut guard = self.preflight.lock().await;
        if let Some(cached) = guard.as_ref() {
            return cached.clone();
        }

        let summary = run_preflight(&self.runtime_root()).await.and_then(|summary| {
            if summary.total_servers == summary.connected_servers
                && summary.total_servers == summary.servers_with_tools
            {
                Ok(summary)
            } else {
                let failures = summary
                    .failures
                    .iter()
                    .map(|failure| {
                        format!(
                            "{}(status={}, has_tools={}, error={})",
                            failure.server_name,
                            failure.connection_status,
                            failure.has_tools,
                            failure.error
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                Err(format!(
                    "MCP-Bench preflight incomplete: connected={}/{} returned_tools={}/{} failures=[{}]",
                    summary.connected_servers,
                    summary.total_servers,
                    summary.servers_with_tools,
                    summary.total_servers,
                    failures
                ))
            }
        });

        *guard = Some(summary.clone());
        summary
    }

    async fn run_native_attempt(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        item: &McpBenchItem,
        ref_answer: &str,
    ) -> Result<Record, String> {
        let mut session = McpBenchSession::connect(&self.runtime_root(), item).await?;
        let available_tools = session.available_tools().clone();
        if available_tools.is_empty() {
            session.shutdown().await;
            return Err("no tools discovered from selected MCP servers".to_string());
        }

        let mut accumulated_information = String::new();
        let mut execution_results = Vec::<McpBenchExecutionResult>::new();
        let mut steps = Vec::<McpBenchStep>::new();
        let mut total_planned_tools = 0usize;
        let mut valid_planned_tools = 0usize;
        let mut executed_rounds = 0usize;

        for round_num in 1..=MCP_BENCH_MAX_ROUNDS {
            let cot_context =
                build_planning_context(item, &available_tools, &accumulated_information);
            let cot = complete_text(
                model_client,
                model_name,
                &cot_context,
                &MCP_BENCH_INFO.sampling_config,
                vec!["</think>".to_string()],
                MCP_BENCH_COT_MAX_TOKENS,
            )
            .await?;

            let decision_prompt = build_planning_decision_prompt(&cot_context, &cot);
            let decision_text = complete_text(
                model_client,
                model_name,
                &decision_prompt,
                &MCP_BENCH_INFO.sampling_config,
                Vec::new(),
                MCP_BENCH_DECISION_MAX_TOKENS,
            )
            .await?;
            let decision = match parse_planning_decision(&decision_text) {
                Ok(decision) => decision,
                Err(err) => {
                    let context = render_debug_context(
                        item,
                        &available_tools,
                        &steps,
                        &accumulated_information,
                        Some(&decision_text),
                        None,
                    );
                    session.shutdown().await;
                    return Ok(Record {
                        context,
                        answer: decision_text,
                        ref_answer: ref_answer.to_string(),
                        is_passed: false,
                        fail_reason: err,
                    });
                }
            };

            let mut round_executions = Vec::<McpBenchExecutionResult>::new();
            if decision.should_continue {
                for (planned_layer, raw_call) in decision.tool_calls.iter().enumerate() {
                    total_planned_tools += 1;
                    match normalize_planned_tool_call(raw_call, &available_tools) {
                        Ok(call) => {
                            valid_planned_tools += 1;
                            round_executions.push(
                                execute_planned_tool(&mut session, &call, round_num, planned_layer)
                                    .await,
                            );
                        }
                        Err(err) => round_executions.push(invalid_planned_tool_execution(
                            raw_call,
                            round_num,
                            planned_layer,
                            err,
                        )),
                    }
                }
            }

            if !round_executions.is_empty() {
                append_round_summary(
                    &mut accumulated_information,
                    round_num,
                    &decision.reasoning,
                    &round_executions,
                );
                execution_results.extend(round_executions.clone());
                executed_rounds = round_num;
            }

            steps.push(McpBenchStep {
                round_num,
                cot,
                decision,
                executions: round_executions.clone(),
            });

            if !steps.last().unwrap().decision.should_continue || round_executions.is_empty() {
                break;
            }
        }

        let planning_json_compliance = if total_planned_tools == 0 {
            1.0
        } else {
            valid_planned_tools as f64 / total_planned_tools as f64
        };

        let final_prompt = build_final_answer_prompt(item, &accumulated_information);
        let final_answer = complete_text(
            model_client,
            model_name,
            &final_prompt,
            &MCP_BENCH_INFO.sampling_config,
            Vec::new(),
            MCP_BENCH_FINAL_MAX_TOKENS,
        )
        .await?;

        let concrete_task_description = if item.task.fuzzy_description.trim().is_empty() {
            String::new()
        } else {
            item.task.task_description.trim().to_string()
        };
        let judge_request = McpBenchJudgeRequest {
            judge_config: resolve_judge_config()?,
            task: super::prompt::presented_task(item).to_string(),
            final_solution: final_answer.trim().to_string(),
            total_rounds: executed_rounds,
            available_tools,
            planning_json_compliance,
            accumulated_information: accumulated_information.clone(),
            concrete_task_description,
            dependency_analysis: item.task.dependency_analysis.trim().to_string(),
            execution_results,
        };
        let evaluation =
            evaluate_with_official_evaluator(&self.runtime_root(), &judge_request).await?;
        let is_passed = collapse_evaluation_to_pass(&evaluation);
        let context = render_debug_context(
            item,
            &judge_request.available_tools,
            &steps,
            &accumulated_information,
            None,
            Some(&final_answer),
        );
        session.shutdown().await;

        Ok(Record {
            context: format!(
                "{}\n\nevaluation={}",
                context,
                summarize_evaluation(&evaluation)
            ),
            answer: final_answer.trim().to_string(),
            ref_answer: ref_answer.to_string(),
            is_passed,
            fail_reason: if is_passed {
                String::new()
            } else {
                summarize_evaluation(&evaluation)
            },
        })
    }

    fn configured_limit() -> Option<usize> {
        std::env::var("RWKV_MCP_BENCH_LIMIT")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|limit| *limit > 0)
    }
}

#[async_trait]
impl Benchmark for McpBench {
    fn load(&mut self) -> bool {
        self.test.clear();
        if !self.runtime_root().join(".git").is_dir() {
            return true;
        }
        if !self.task_assets_root().is_dir() {
            return true;
        }

        self.test = self
            .load_items()
            .unwrap_or_else(|err| panic!("failed to load mcp_bench: {err}"));
        if let Some(limit) = Self::configured_limit() {
            self.test.truncate(limit.min(self.test.len()));
        }
        if let Ok(mut guard) = self.preflight.try_lock() {
            *guard = None;
        }
        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let dataset_root = mcp_bench_root(&self.dataset_root);
        let runtime_root = self.runtime_root();
        let tasks_root = self.task_assets_root();

        if !dataset_root.is_dir() || !runtime_root.is_dir() || !tasks_root.is_dir() {
            return true;
        }
        match check_runtime_checkout(&runtime_root) {
            Ok(true) => {}
            Ok(false) | Err(_) => return true,
        }

        let required_runtime_files = [
            runtime_root.join("benchmark").join("runner.py"),
            runtime_root.join("benchmark").join("evaluator.py"),
            runtime_root.join("utils").join("collect_mcp_info.py"),
            runtime_root.join("utils").join("local_server_config.py"),
            runtime_root.join("mcp_servers").join("commands.json"),
        ];
        if required_runtime_files.iter().any(|path| !path.is_file()) {
            return true;
        }
        if MCP_BENCH_TASK_FILES
            .iter()
            .any(|file_name| !tasks_root.join(file_name).is_file())
        {
            return true;
        }

        let expected_len = Self::configured_limit()
            .map(|limit| limit.min(MCP_BENCH_EXPECTED_LEN))
            .unwrap_or(MCP_BENCH_EXPECTED_LEN);

        self.test.len() != expected_len
            || self.test.iter().any(|item| {
                item.server_name.trim().is_empty()
                    || item.task.task_id.trim().is_empty()
                    || item.task.task_description.trim().is_empty()
            })
    }

    async fn download(&self) {
        ensure_runtime_checkout(&self.dataset_root)
            .unwrap_or_else(|err| panic!("failed to prepare MCP-Bench runtime: {err}"));
        sync_task_assets(&self.dataset_root)
            .unwrap_or_else(|err| panic!("failed to sync MCP-Bench task assets: {err}"));
        println!("mcp_bench dataset: {}", self.runtime_root().display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(cot_mode, CoTMode::CoT, "mcp_bench only supports CoT");
        assert_eq!(n_shot, 0, "mcp_bench only supports 0-shot");
        build_context_summary(&self.test[index])
    }

    fn get_ref_answer(&self, index: usize) -> String {
        let item = &self.test[index];
        format!(
            concat!(
                "task_id={}\n",
                "task_file={}\n",
                "server_name={}\n",
                "servers={}\n",
                "combination_type={}\n",
                "runtime_commit={}\n",
                "evaluator=official_mcp_bench_evaluator_phase2"
            ),
            item.task.task_id,
            item.task_file,
            item.server_name,
            item.servers.join(", "),
            item.combination_type,
            MCP_BENCH_GIT_COMMIT,
        )
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        _judger_model_name: Option<&str>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let context = self.get_expected_context(index, CoTMode::CoT, 0);
        let ref_answer = self.get_ref_answer(index);

        if cot_mode != CoTMode::CoT || n_shot != 0 {
            return Record {
                context,
                answer: String::new(),
                ref_answer,
                is_passed: false,
                fail_reason: format!(
                    "unsupported mcp_bench config: cot_mode={cot_mode:?}, n_shot={n_shot}"
                ),
            };
        }

        let preflight = match self.ensure_preflight_once().await {
            Ok(summary) => summary,
            Err(err) => {
                return Record {
                    context,
                    answer: String::new(),
                    ref_answer,
                    is_passed: false,
                    fail_reason: err,
                };
            }
        };

        let item = &self.test[index];
        let mut last_infra_error = String::new();

        for attempt in 0..MCP_BENCH_TASK_RETRY_MAX {
            let attempt_result = timeout(
                Duration::from_secs(MCP_BENCH_TASK_TIMEOUT_SECS),
                self.run_native_attempt(model_name, model_client, item, &ref_answer),
            )
            .await;

            match attempt_result {
                Ok(Ok(record)) => return record,
                Ok(Err(err)) => last_infra_error = err,
                Err(_) => {
                    last_infra_error = format!(
                        "mcp_bench native execution timed out after {} seconds",
                        MCP_BENCH_TASK_TIMEOUT_SECS
                    );
                }
            }

            if attempt + 1 < MCP_BENCH_TASK_RETRY_MAX {
                sleep(Duration::from_secs(MCP_BENCH_RETRY_DELAY_SECS)).await;
            }
        }

        let preflight_summary = if preflight.captured_stdout.trim().is_empty() {
            String::new()
        } else {
            format!("\npreflight_stdout={}", preflight.captured_stdout.trim())
        };

        Record {
            context,
            answer: String::new(),
            ref_answer,
            is_passed: false,
            fail_reason: format!("{last_infra_error}{preflight_summary}"),
        }
    }
}

async fn execute_planned_tool(
    session: &mut McpBenchSession,
    call: &PlannedToolCall,
    round_num: usize,
    planned_layer: usize,
) -> McpBenchExecutionResult {
    let full_tool_name = call.full_name();
    match session
        .call_tool(&full_tool_name, &object_to_value(&call.arguments))
        .await
    {
        Ok((true, text)) => McpBenchExecutionResult {
            tool: full_tool_name,
            server: call.server.clone(),
            parameters: call.arguments.clone(),
            round_num,
            planned_layer: Some(planned_layer),
            success: true,
            result: Some(text),
            error: None,
        },
        Ok((false, text)) => McpBenchExecutionResult {
            tool: full_tool_name,
            server: call.server.clone(),
            parameters: call.arguments.clone(),
            round_num,
            planned_layer: Some(planned_layer),
            success: false,
            result: None,
            error: Some(text),
        },
        Err(err) => McpBenchExecutionResult {
            tool: full_tool_name,
            server: call.server.clone(),
            parameters: call.arguments.clone(),
            round_num,
            planned_layer: Some(planned_layer),
            success: false,
            result: None,
            error: Some(err),
        },
    }
}

fn invalid_planned_tool_execution(
    call: &PlannedToolCall,
    round_num: usize,
    planned_layer: usize,
    error: String,
) -> McpBenchExecutionResult {
    let tool_name = if call.server.trim().is_empty() {
        call.tool.clone()
    } else {
        call.full_name()
    };
    McpBenchExecutionResult {
        tool: tool_name,
        server: if call.server.trim().is_empty() {
            "unknown".to_string()
        } else {
            call.server.clone()
        },
        parameters: call.arguments.clone(),
        round_num,
        planned_layer: Some(planned_layer),
        success: false,
        result: None,
        error: Some(error),
    }
}

fn render_debug_context(
    item: &McpBenchItem,
    available_tools: &std::collections::BTreeMap<String, super::types::McpBenchAvailableTool>,
    steps: &[McpBenchStep],
    accumulated_information: &str,
    last_decision_response: Option<&str>,
    final_answer: Option<&str>,
) -> String {
    let tool_names = available_tools
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    let mut text = build_context_summary(item);
    text.push_str(&format!(
        "\navailable_tools_count={}\navailable_tools:\n{}\n",
        available_tools.len(),
        tool_names
    ));
    if !steps.is_empty() {
        text.push_str("\ntrace:\n");
        text.push_str(&render_trace(steps));
        text.push('\n');
    }
    if !accumulated_information.trim().is_empty() {
        text.push_str("\naccumulated_information:\n");
        text.push_str(accumulated_information.trim());
        text.push('\n');
    }
    if let Some(last_decision_response) = last_decision_response {
        text.push_str("\nlast_decision_response:\n");
        text.push_str(last_decision_response.trim());
        text.push('\n');
    }
    if let Some(final_answer) = final_answer {
        text.push_str("\nfinal_answer:\n");
        text.push_str(final_answer.trim());
        text.push('\n');
    }
    text
}

fn object_to_value(object: &sonic_rs::Object) -> Value {
    sonic_rs::from_str::<Value>(&sonic_rs::to_string(object).unwrap_or_else(|_| "{}".to_string()))
        .unwrap_or_else(|_| json!({}))
}

async fn complete_text(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt: &str,
    sampling_config: &SamplingConfig,
    stop: Vec<String>,
    max_tokens: u32,
) -> Result<String, String> {
    generate_text_completion(
        model_client,
        model_name,
        prompt,
        stop,
        max_tokens,
        sampling_config,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::McpBench;
    use crate::datasets::function_calling::mcp_bench::prompt::build_context_summary;
    use crate::datasets::function_calling::mcp_bench::types::RawTaskFile;

    #[test]
    fn flatten_runner_format_groups() {
        let raw = r#"{
          "server_tasks": [
            {
              "server_name": "Server A",
              "servers": ["Server A"],
              "combination_name": "single_server_a",
              "combination_type": "single",
              "tasks": [
                {
                  "task_id": "a_000",
                  "task_description": "detailed task",
                  "fuzzy_description": "fuzzy task",
                  "dependency_analysis": "deps",
                  "distraction_servers": ["Time MCP"]
                }
              ]
            }
          ]
        }"#;
        let parsed = sonic_rs::from_str::<RawTaskFile>(raw).unwrap();
        let group = &parsed.server_tasks[0];
        let item = build_context_summary(&super::McpBenchItem {
            task_file: "mcpbench_tasks_single_runner_format.json".to_string(),
            server_name: group.server_name.clone(),
            combination_name: group.combination_name.clone(),
            combination_type: group.combination_type.clone(),
            servers: group.servers.clone(),
            task: group.tasks[0].clone(),
        });
        assert!(item.contains("phase=2_rust_native_completion_loop"));
        assert!(item.contains("task_id=a_000"));
        let _ = McpBench::new("/tmp");
    }
}
