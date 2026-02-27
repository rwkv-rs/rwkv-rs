use std::collections::{BTreeMap, VecDeque};
use std::error::Error as _;
use std::io::IsTerminal;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::{StreamExt, stream::FuturesUnordered};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tokio::sync::{Semaphore, mpsc};

use crate::metrics::{AggregateMetrics, RequestMetrics, StageBreakdownMs, aggregate_metrics};
use crate::{BenchError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Endpoint {
    Completions,
    ChatCompletions,
}

impl Endpoint {
    fn path(self) -> &'static str {
        match self {
            Endpoint::Completions => "/v1/completions",
            Endpoint::ChatCompletions => "/v1/chat/completions",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeConfig {
    pub base_url: String,
    pub model: String,
    pub endpoint: Endpoint,
    pub num_requests: usize,
    pub concurrency: usize,
    pub request_rate: f64,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_penalty_decay")]
    pub penalty_decay: f32,
    pub timeout_secs: u64,
    pub api_key: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServeRunResult {
    pub started_at_utc: String,
    pub finished_at_utc: String,
    pub config: ServeConfig,
    pub duration_s: f64,
    pub summary: AggregateMetrics,
    pub requests: Vec<RequestMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepConfig {
    pub base: ServeConfig,
    pub request_counts: Vec<usize>,
    pub before_each_command: Option<String>,
    pub after_each_command: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepCaseResult {
    pub request_count: usize,
    pub run: ServeRunResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepBestCase {
    pub request_count: usize,
    pub total_token_throughput: f64,
    pub request_throughput: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepRunResult {
    pub started_at_utc: String,
    pub finished_at_utc: String,
    pub config: SweepConfig,
    pub cases: Vec<SweepCaseResult>,
    pub best_case: Option<SweepBestCase>,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> i32 {
    500
}

fn default_top_p() -> f32 {
    0.3
}

fn default_presence_penalty() -> f32 {
    0.5
}

fn default_repetition_penalty() -> f32 {
    0.5
}

fn default_penalty_decay() -> f32 {
    0.996
}

struct TokenEvent {
    at: Instant,
}

#[derive(Default)]
struct LiveStats {
    processed: usize,
    completed: usize,
    failed: usize,
    output_tokens: usize,
    e2el_sum_s: f64,
    first_error: Option<String>,
    token_window: VecDeque<Instant>,
}

impl LiveStats {
    fn observe(&mut self, metrics: &RequestMetrics) {
        self.processed += 1;
        if metrics.success {
            self.completed += 1;
            self.output_tokens += metrics.output_tokens;
        } else {
            self.failed += 1;
            if self.first_error.is_none() {
                self.first_error = metrics.error.clone();
            }
        }
        self.e2el_sum_s += metrics.e2el_s.max(0.0);
    }

    fn record_token(&mut self, at: Instant) {
        self.token_window.push_back(at);
    }

    fn instant_tok_s(&mut self, window_secs: f64) -> f64 {
        let cutoff = Instant::now() - Duration::from_secs_f64(window_secs);
        while self.token_window.front().map_or(false, |t| *t < cutoff) {
            self.token_window.pop_front();
        }
        self.token_window.len() as f64 / window_secs
    }

    fn status_line(&self, elapsed_s: f64, inflight: usize, inst_tok_s: f64) -> String {
        let safe_elapsed = elapsed_s.max(1e-9);
        let success_rps = self.completed as f64 / safe_elapsed;
        let mean_e2el_ms = if self.processed > 0 {
            self.e2el_sum_s * 1000.0 / self.processed as f64
        } else {
            0.0
        };

        format!(
            "ok={} fail={} inflight={} rps={success_rps:.1} tok/s={inst_tok_s:.1} e2el={mean_e2el_ms:.1}ms",
            self.completed, self.failed, inflight
        )
    }
}

fn create_progress_bar(total_requests: usize) -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }

    let bar = ProgressBar::new(total_requests as u64);
    let style = match ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} req {msg}",
    ) {
        Ok(style) => style.progress_chars("#>-"),
        Err(_) => ProgressStyle::default_bar(),
    };
    bar.set_style(style);
    bar.enable_steady_tick(Duration::from_millis(120));
    bar
}

fn format_elapsed(elapsed: Duration) -> String {
    let total = elapsed.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn build_prompt(input_tokens: usize, request_id: usize) -> String {
    let repeat = input_tokens.max(1);
    let mut prompt = format!("request-{} ", request_id);
    for _ in 0..repeat {
        prompt.push_str("hello ");
    }
    prompt
}

fn endpoint_url(cfg: &ServeConfig) -> String {
    format!(
        "{}{}",
        cfg.base_url.trim_end_matches('/'),
        cfg.endpoint.path()
    )
}

fn format_reqwest_error(context: &str, err: &reqwest::Error) -> String {
    let mut message = format!("{context}: {err}");

    if err.is_connect() {
        message.push_str(" [kind=connect]");
    }
    if err.is_timeout() {
        message.push_str(" [kind=timeout]");
    }
    if err.is_decode() {
        message.push_str(" [kind=decode]");
    }
    if let Some(url) = err.url() {
        message.push_str(&format!(" [url={url}]"));
    }

    let mut source = err.source();
    let mut depth = 0usize;
    while let Some(cause) = source {
        depth += 1;
        message.push_str(&format!(" | source[{depth}]: {cause}"));
        source = cause.source();
    }

    message
}

fn extract_chunk_text_from_value(value: &serde_json::Value) -> String {
    if let Some(choices) = value.get("choices").and_then(serde_json::Value::as_array)
        && let Some(choice) = choices.first()
    {
        if let Some(text) = choice.get("text").and_then(serde_json::Value::as_str) {
            return text.to_string();
        }

        if let Some(message) = choice.get("message")
            && let Some(content) = message.get("content").and_then(serde_json::Value::as_str)
        {
            return content.to_string();
        }
    }
    String::new()
}

fn has_finish_reason(value: &serde_json::Value) -> bool {
    value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(serde_json::Value::as_str)
        .is_some()
}

fn parse_timings_ms(value: &serde_json::Value) -> StageBreakdownMs {
    let timings = value.get("timings_ms").cloned().unwrap_or_default();
    StageBreakdownMs {
        validate_ms: read_from(&timings, "validate_ms"),
        tokenize_ms: read_from(&timings, "tokenize_ms"),
        queue_wait_ms: read_from(&timings, "queue_wait_ms"),
        schedule_wait_ms: read_from(&timings, "schedule_wait_ms"),
        prefill_first_ms: read_from(&timings, "prefill_first_ms"),
        first_emit_ms: read_from(&timings, "first_emit_ms"),
        prefill_total_ms: read_from(&timings, "prefill_total_ms"),
        decode_total_ms: read_from(&timings, "decode_total_ms"),
        request_total_ms: read_from(&timings, "request_total_ms"),
    }
}

fn read_from(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key).and_then(serde_json::Value::as_f64)
}

fn parse_sse_line(
    line: &str,
    bench_start: Instant,
    first_token_s: &mut Option<f64>,
    last_token_s: &mut Option<f64>,
    itl_s: &mut Vec<f64>,
    output_tokens: &mut usize,
    stage_ms: &mut StageBreakdownMs,
    saw_terminal: &mut bool,
) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return false;
    }

    let payload = trimmed.trim_start_matches("data:").trim();
    if payload.is_empty() {
        return false;
    }

    if payload == "[DONE]" {
        *saw_terminal = true;
        return true;
    }

    let value = match serde_json::from_str::<serde_json::Value>(payload) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let text = extract_chunk_text_from_value(&value);
    if text.is_empty() {
        if has_finish_reason(&value) {
            *stage_ms = parse_timings_ms(&value);
            *saw_terminal = true;
            return true;
        }
        return false;
    }

    let now_s = bench_start.elapsed().as_secs_f64();
    if first_token_s.is_none() {
        *first_token_s = Some(now_s);
    } else if let Some(last) = *last_token_s {
        itl_s.push((now_s - last).max(0.0));
    }

    *last_token_s = Some(now_s);
    *output_tokens += 1;
    false
}

async fn run_single_request(
    client: reqwest::Client,
    cfg: ServeConfig,
    request_id: usize,
    token_tx: Option<mpsc::UnboundedSender<TokenEvent>>,
) -> RequestMetrics {
    let prompt = build_prompt(cfg.input_tokens, request_id);
    let start = Instant::now();

    let payload = match cfg.endpoint {
        Endpoint::Completions => serde_json::json!({
            "model": cfg.model,
            "prompt": prompt,
            "stream": cfg.stream,
            "max_tokens": cfg.output_tokens,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "presence_penalty": cfg.presence_penalty,
            "repetition_penalty": cfg.repetition_penalty,
            "penalty_decay": cfg.penalty_decay,
        }),
        Endpoint::ChatCompletions => serde_json::json!({
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": cfg.stream,
            "max_tokens": cfg.output_tokens,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "presence_penalty": cfg.presence_penalty,
            "repetition_penalty": cfg.repetition_penalty,
            "penalty_decay": cfg.penalty_decay,
        }),
    };

    let mut request = client
        .post(endpoint_url(&cfg))
        .header("Content-Type", "application/json")
        .json(&payload);

    if let Some(key) = &cfg.api_key {
        request = request.header("Authorization", format!("Bearer {key}"));
    }

    let response = request.send().await;
    let mut metrics = RequestMetrics {
        request_id,
        success: false,
        error: None,
        prompt_tokens: cfg.input_tokens,
        output_tokens: 0,
        ttft_s: None,
        itl_s: Vec::new(),
        tpot_s: None,
        e2el_s: 0.0,
        stage_ms: StageBreakdownMs::default(),
    };

    let response = match response {
        Ok(resp) => resp,
        Err(err) => {
            let prefix = if err.is_timeout() {
                "timeout"
            } else {
                "request failed"
            };
            metrics.error = Some(format_reqwest_error(prefix, &err));
            metrics.e2el_s = start.elapsed().as_secs_f64();
            return metrics;
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = match response.text().await {
            Ok(body) => body,
            Err(err) => format!(
                "<failed to read body: {}>",
                format_reqwest_error("body read failed", &err)
            ),
        };
        metrics.error = Some(format!("http {}: {}", status.as_u16(), body));
        metrics.e2el_s = start.elapsed().as_secs_f64();
        return metrics;
    }

    if cfg.stream {
        let content_encoding = response
            .headers()
            .get(reqwest::header::CONTENT_ENCODING)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let mut byte_stream = response.bytes_stream();
        let mut sse_buffer = String::new();
        let mut first_token_s = None;
        let mut last_token_s = None;
        let mut saw_terminal = false;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(err) => {
                    let mut msg = format_reqwest_error("stream read failed", &err);
                    msg.push_str(&format!(
                        " [content-encoding={}]",
                        content_encoding.as_deref().unwrap_or("none")
                    ));
                    metrics.error = Some(msg);
                    break;
                }
            };

            sse_buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_idx) = sse_buffer.find('\n') {
                let line = sse_buffer[..newline_idx].trim_end_matches('\r').to_string();
                sse_buffer.replace_range(..=newline_idx, "");
                let prev_tokens = metrics.output_tokens;
                if parse_sse_line(
                    &line,
                    start,
                    &mut first_token_s,
                    &mut last_token_s,
                    &mut metrics.itl_s,
                    &mut metrics.output_tokens,
                    &mut metrics.stage_ms,
                    &mut saw_terminal,
                ) {
                    metrics.success = true;
                }
                if metrics.output_tokens > prev_tokens {
                    if let Some(tx) = &token_tx {
                        let _ = tx.send(TokenEvent { at: Instant::now() });
                    }
                }
            }
        }

        if metrics.error.is_none() && !sse_buffer.trim().is_empty() {
            for line in sse_buffer.lines() {
                let prev_tokens = metrics.output_tokens;
                if parse_sse_line(
                    line,
                    start,
                    &mut first_token_s,
                    &mut last_token_s,
                    &mut metrics.itl_s,
                    &mut metrics.output_tokens,
                    &mut metrics.stage_ms,
                    &mut saw_terminal,
                ) {
                    metrics.success = true;
                }
                if metrics.output_tokens > prev_tokens {
                    if let Some(tx) = &token_tx {
                        let _ = tx.send(TokenEvent { at: Instant::now() });
                    }
                }
            }
        }

        metrics.ttft_s = first_token_s.map(|t| t.max(0.0));
        metrics.e2el_s = start.elapsed().as_secs_f64();
        if metrics.output_tokens > 1
            && let Some(ttft) = metrics.ttft_s
        {
            metrics.tpot_s = Some((metrics.e2el_s - ttft) / (metrics.output_tokens as f64 - 1.0));
        }

        if metrics.error.is_none() && saw_terminal {
            metrics.success = true;
        } else if metrics.error.is_none() {
            metrics.error = Some("stream ended without terminal event".to_string());
            metrics.success = false;
        }
        return metrics;
    }

    let body = response.text().await;
    match body {
        Ok(raw) => {
            let parsed = serde_json::from_str::<serde_json::Value>(&raw);
            if let Ok(value) = parsed {
                metrics.stage_ms = parse_timings_ms(&value);
                let text = extract_chunk_text_from_value(&value);
                metrics.output_tokens = text.split_whitespace().count();
            }

            metrics.e2el_s = start.elapsed().as_secs_f64();
            metrics.ttft_s = Some(metrics.e2el_s);
            metrics.tpot_s = Some(0.0);
            metrics.success = true;
            metrics
        }
        Err(err) => {
            metrics.error = Some(format_reqwest_error("response decode failed", &err));
            metrics.e2el_s = start.elapsed().as_secs_f64();
            metrics
        }
    }
}

pub async fn run_serve_benchmark(cfg: ServeConfig) -> Result<ServeRunResult> {
    let started_at_utc = Utc::now().to_rfc3339();
    let bench_start = Instant::now();
    let progress_visible = std::io::stderr().is_terminal();

    let mut client_builder = reqwest::Client::builder();
    if cfg.timeout_secs > 0 {
        client_builder = client_builder.timeout(Duration::from_secs(cfg.timeout_secs));
    }
    let client = client_builder.build()?;

    let semaphore = Arc::new(Semaphore::new(cfg.concurrency.max(1)));
    let mut inflight = FuturesUnordered::new();
    let progress = create_progress_bar(cfg.num_requests);
    let mut live_stats = LiveStats::default();
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_text_update = Instant::now();
    let (token_tx, mut token_rx) = mpsc::unbounded_channel::<TokenEvent>();
    let mut submitted = 0usize;

    for request_id in 0..cfg.num_requests {
        if cfg.request_rate.is_finite() && cfg.request_rate > 0.0 {
            let scheduled = request_id as f64 / cfg.request_rate;
            let elapsed = bench_start.elapsed().as_secs_f64();
            if scheduled > elapsed {
                tokio::time::sleep(Duration::from_secs_f64(scheduled - elapsed)).await;
            }
        }

        let client = client.clone();
        let cfg_cloned = cfg.clone();
        let semaphore = semaphore.clone();
        let tx = token_tx.clone();
        inflight.push(async move {
            let metrics = match semaphore.acquire_owned().await {
                Ok(permit) => {
                    let _permit = permit;
                    run_single_request(client, cfg_cloned, request_id, Some(tx)).await
                }
                Err(err) => RequestMetrics {
                    request_id,
                    success: false,
                    error: Some(format!("acquire error: {err}")),
                    prompt_tokens: cfg_cloned.input_tokens,
                    output_tokens: 0,
                    ttft_s: None,
                    itl_s: Vec::new(),
                    tpot_s: None,
                    e2el_s: 0.0,
                    stage_ms: StageBreakdownMs::default(),
                },
            };
            (request_id, metrics)
        });
        submitted += 1;
    }
    // Drop the original sender so the channel closes when all tasks finish
    drop(token_tx);

    let mut requests = Vec::with_capacity(cfg.num_requests);
    while !inflight.is_empty() {
        tokio::select! {
            biased;
            _ = tick.tick() => {
                while let Ok(ev) = token_rx.try_recv() {
                    live_stats.record_token(ev.at);
                }
                let inst_tok_s = live_stats.instant_tok_s(5.0);
                let inflight_count = submitted.saturating_sub(live_stats.processed);
                let status = live_stats.status_line(
                    bench_start.elapsed().as_secs_f64(),
                    inflight_count,
                    inst_tok_s,
                );
                progress.set_message(status.clone());
                progress.tick();

                if !progress_visible
                    && last_text_update.elapsed() >= Duration::from_secs(2)
                {
                    eprintln!(
                        "[{}] {}/{} req {}",
                        format_elapsed(bench_start.elapsed()),
                        live_stats.processed,
                        cfg.num_requests,
                        status
                    );
                    last_text_update = Instant::now();
                }
            }
            maybe = inflight.next() => {
                let Some((_request_id, metrics)) = maybe else {
                    break;
                };
                live_stats.observe(&metrics);
                progress.inc(1);
                let inflight_count = submitted.saturating_sub(live_stats.processed);
                let inst_tok_s = live_stats.instant_tok_s(5.0);
                progress.set_message(live_stats.status_line(
                    bench_start.elapsed().as_secs_f64(),
                    inflight_count,
                    inst_tok_s,
                ));
                requests.push(metrics);
            }
        }
    }

    requests.sort_by_key(|request| request.request_id);

    let duration_s = bench_start.elapsed().as_secs_f64();
    let final_tok_s = live_stats.instant_tok_s(5.0);
    progress.finish_with_message(format!("done {}", live_stats.status_line(duration_s, 0, final_tok_s)));
    if let Some(err) = live_stats.first_error.as_deref() {
        eprintln!("first request error: {err}");
    }
    let summary = aggregate_metrics(&requests, duration_s);

    Ok(ServeRunResult {
        started_at_utc,
        finished_at_utc: Utc::now().to_rfc3339(),
        config: cfg,
        duration_s,
        summary,
        requests,
    })
}

fn render_template(template: &str, request_count: usize) -> String {
    template
        .replace("{request_count}", &request_count.to_string())
        .replace("{concurrency}", &request_count.to_string())
}

fn maybe_run_template(template: &Option<String>, request_count: usize) -> Result<()> {
    let Some(template) = template else {
        return Ok(());
    };

    let rendered = render_template(template, request_count);
    let status = Command::new("bash").arg("-lc").arg(&rendered).status()?;
    if !status.success() {
        return Err(BenchError::command_failed(rendered, status.code()));
    }
    Ok(())
}

pub async fn run_sweep(cfg: SweepConfig) -> Result<SweepRunResult> {
    let mut cases = Vec::new();
    let started_at_utc = Utc::now().to_rfc3339();

    for &request_count in &cfg.request_counts {
        maybe_run_template(&cfg.before_each_command, request_count)?;

        let mut case_cfg = cfg.base.clone();
        case_cfg.num_requests = request_count;
        case_cfg.concurrency = request_count;
        case_cfg.request_rate = 0.0;
        case_cfg
            .metadata
            .insert("request_count".to_string(), request_count.to_string());
        case_cfg
            .metadata
            .insert("concurrency".to_string(), request_count.to_string());

        let run = run_serve_benchmark(case_cfg).await?;
        cases.push(SweepCaseResult { request_count, run });

        maybe_run_template(&cfg.after_each_command, request_count)?;
    }

    let best_case = cases
        .iter()
        .max_by(|a, b| {
            a.run
                .summary
                .total_token_throughput
                .total_cmp(&b.run.summary.total_token_throughput)
        })
        .map(|best| SweepBestCase {
            request_count: best.request_count,
            total_token_throughput: best.run.summary.total_token_throughput,
            request_throughput: best.run.summary.request_throughput,
        });

    Ok(SweepRunResult {
        started_at_utc,
        finished_at_utc: Utc::now().to_rfc3339(),
        config: cfg,
        cases,
        best_case,
    })
}

pub fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}
