use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::metrics::{AggregateMetrics, RequestMetrics, aggregate_metrics};
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
    pub temperature: f32,
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
    pub batch_sizes: Vec<usize>,
    pub paragraph_lens: Vec<usize>,
    pub backends: Vec<String>,
    pub before_each_command: Option<String>,
    pub after_each_command: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepCaseResult {
    pub backend: String,
    pub batch_size: usize,
    pub paragraph_len: usize,
    pub run: ServeRunResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepBestCase {
    pub backend: String,
    pub batch_size: usize,
    pub paragraph_len: usize,
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

fn extract_chunk_text(chunk: &str) -> String {
    let parsed = serde_json::from_str::<serde_json::Value>(chunk);
    if let Ok(value) = parsed
        && let Some(choices) = value.get("choices").and_then(serde_json::Value::as_array)
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

fn parse_sse_line(
    line: &str,
    bench_start: Instant,
    first_token_s: &mut Option<f64>,
    last_token_s: &mut Option<f64>,
    itl_s: &mut Vec<f64>,
    output_tokens: &mut usize,
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
        return true;
    }

    let text = extract_chunk_text(payload);
    if text.is_empty() {
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
        }),
        Endpoint::ChatCompletions => serde_json::json!({
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": cfg.stream,
            "max_tokens": cfg.output_tokens,
            "temperature": cfg.temperature,
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
    };

    let response = match response {
        Ok(resp) => resp,
        Err(err) => {
            metrics.error = Some(format!("request failed: {err}"));
            metrics.e2el_s = start.elapsed().as_secs_f64();
            return metrics;
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<failed to read body>".to_string());
        metrics.error = Some(format!("http {}: {}", status.as_u16(), body));
        metrics.e2el_s = start.elapsed().as_secs_f64();
        return metrics;
    }

    if cfg.stream {
        let mut byte_stream = response.bytes_stream();
        let mut sse_buffer = String::new();
        let mut first_token_s = None;
        let mut last_token_s = None;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(err) => {
                    metrics.error = Some(format!("stream read failed: {err}"));
                    break;
                }
            };

            sse_buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_idx) = sse_buffer.find('\n') {
                let line = sse_buffer[..newline_idx].trim_end_matches('\r').to_string();
                sse_buffer.replace_range(..=newline_idx, "");
                if parse_sse_line(
                    &line,
                    start,
                    &mut first_token_s,
                    &mut last_token_s,
                    &mut metrics.itl_s,
                    &mut metrics.output_tokens,
                ) {
                    metrics.success = true;
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

        if metrics.error.is_none() {
            metrics.success = true;
        }
        return metrics;
    }

    let body = response.text().await;
    match body {
        Ok(raw) => {
            let parsed = serde_json::from_str::<serde_json::Value>(&raw);
            if let Ok(value) = parsed {
                let text = value
                    .get("choices")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|choices| choices.first())
                    .and_then(|choice| choice.get("text"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                metrics.output_tokens = text.split_whitespace().count();
            }

            metrics.e2el_s = start.elapsed().as_secs_f64();
            metrics.ttft_s = Some(metrics.e2el_s);
            metrics.tpot_s = Some(0.0);
            metrics.success = true;
            metrics
        }
        Err(err) => {
            metrics.error = Some(format!("response decode failed: {err}"));
            metrics.e2el_s = start.elapsed().as_secs_f64();
            metrics
        }
    }
}

pub async fn run_serve_benchmark(cfg: ServeConfig) -> Result<ServeRunResult> {
    let started_at_utc = Utc::now().to_rfc3339();
    let bench_start = Instant::now();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
        .build()?;

    let semaphore = Arc::new(Semaphore::new(cfg.concurrency.max(1)));
    let mut handles = Vec::with_capacity(cfg.num_requests);

    for request_id in 0..cfg.num_requests {
        if cfg.request_rate.is_finite() && cfg.request_rate > 0.0 {
            let scheduled = request_id as f64 / cfg.request_rate;
            let elapsed = bench_start.elapsed().as_secs_f64();
            if scheduled > elapsed {
                tokio::time::sleep(Duration::from_secs_f64(scheduled - elapsed)).await;
            }
        }

        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let cfg_cloned = cfg.clone();
        let handle = tokio::spawn(async move {
            let _permit = permit;
            run_single_request(client, cfg_cloned, request_id).await
        });
        handles.push((request_id, handle));
    }

    let mut requests = Vec::with_capacity(cfg.num_requests);
    for (request_id, handle) in handles {
        match handle.await {
            Ok(metrics) => requests.push(metrics),
            Err(err) => requests.push(RequestMetrics {
                request_id,
                success: false,
                error: Some(format!("join error: {err}")),
                prompt_tokens: cfg.input_tokens,
                output_tokens: 0,
                ttft_s: None,
                itl_s: Vec::new(),
                tpot_s: None,
                e2el_s: 0.0,
            }),
        }
    }

    requests.sort_by_key(|request| request.request_id);

    let duration_s = bench_start.elapsed().as_secs_f64();
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

fn render_template(
    template: &str,
    backend: &str,
    batch_size: usize,
    paragraph_len: usize,
) -> String {
    template
        .replace("{backend}", backend)
        .replace("{batch_size}", &batch_size.to_string())
        .replace("{paragraph_len}", &paragraph_len.to_string())
}

fn maybe_run_template(
    template: &Option<String>,
    backend: &str,
    batch_size: usize,
    paragraph_len: usize,
) -> Result<()> {
    let Some(template) = template else {
        return Ok(());
    };

    let rendered = render_template(template, backend, batch_size, paragraph_len);
    let status = Command::new("bash").arg("-lc").arg(&rendered).status()?;
    if !status.success() {
        return Err(BenchError::command_failed(rendered, status.code()));
    }
    Ok(())
}

pub async fn run_sweep(cfg: SweepConfig) -> Result<SweepRunResult> {
    let mut cases = Vec::new();
    let started_at_utc = Utc::now().to_rfc3339();

    for backend in &cfg.backends {
        for &batch_size in &cfg.batch_sizes {
            for &paragraph_len in &cfg.paragraph_lens {
                maybe_run_template(&cfg.before_each_command, backend, batch_size, paragraph_len)?;

                let mut case_cfg = cfg.base.clone();
                case_cfg
                    .metadata
                    .insert("backend".to_string(), backend.clone());
                case_cfg
                    .metadata
                    .insert("batch_size".to_string(), batch_size.to_string());
                case_cfg
                    .metadata
                    .insert("paragraph_len".to_string(), paragraph_len.to_string());

                let run = run_serve_benchmark(case_cfg).await?;
                cases.push(SweepCaseResult {
                    backend: backend.clone(),
                    batch_size,
                    paragraph_len,
                    run,
                });

                maybe_run_template(&cfg.after_each_command, backend, batch_size, paragraph_len)?;
            }
        }
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
            backend: best.backend.clone(),
            batch_size: best.batch_size,
            paragraph_len: best.paragraph_len,
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
