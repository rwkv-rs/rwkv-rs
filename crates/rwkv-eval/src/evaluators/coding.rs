use std::sync::atomic::{AtomicU64, Ordering};

use async_openai::{Client, config::OpenAIConfig};
use microsandbox::sandbox::Sandbox;
use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::{
    datasets::SamplingConfig,
    inferers::{CompletionRequest, CompletionResponse},
};

static SANDBOX_COUNTER: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(1));

const DEFAULT_MEMORY_MB: u32 = 512;
const DEFAULT_CPUS: u8 = 1;

#[derive(Debug, Clone)]
pub struct SandboxVerdict {
    pub passed: bool,
    pub fail_reason: String,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Deserialize)]
struct SandboxVerdictWire {
    passed: bool,
    fail_reason: String,
}

pub async fn get_completion(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt: &str,
    sampling_config: &SamplingConfig,
    stop: Vec<String>,
    max_tokens: u32,
) -> String {
    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt.into(),
        stop,
        max_tokens,
        sampling_config,
        None,
        None,
    );
    let resp: CompletionResponse = model_client.completions().create_byot(&req).await.unwrap();
    resp.choices[0].text.clone()
}

pub async fn ensure_microsandbox_available() -> Result<(), String> {
    let probe_script = r#"
import json
print(json.dumps({"passed": True, "fail_reason": ""}))
"#;

    let verdict = run_python_verdict_script(probe_script)
        .await
        .map_err(|err| {
            format!("microsandbox unavailable: {err}. start it first with `msb server start --dev`")
        })?;

    if verdict.passed {
        Ok(())
    } else {
        Err(format!(
            "microsandbox probe failed: {}. start it first with `msb server start --dev`",
            verdict.fail_reason
        ))
    }
}

pub async fn run_python_verdict_script(script: &str) -> Result<SandboxVerdict, String> {
    let name = next_sandbox_name();
    let sandbox = Sandbox::builder(name.clone())
        .image("python:3.12")
        .memory(DEFAULT_MEMORY_MB)
        .cpus(DEFAULT_CPUS)
        .create()
        .await
        .map_err(|err| format!("create sandbox `{name}` failed: {err}"))?;

    let execution = sandbox.exec("python", ["-c", script]).await;
    let stop_result = sandbox.stop().await;

    if let Err(err) = stop_result {
        eprintln!("failed to stop microsandbox `{name}`: {err}");
    } else if let Err(err) = sandbox.wait().await {
        eprintln!("failed to wait for microsandbox `{name}` shutdown: {err}");
    }

    match execution {
        Ok(execution) => {
            let stdout = String::from_utf8_lossy(execution.stdout_bytes()).into_owned();
            let stderr = String::from_utf8_lossy(execution.stderr_bytes()).into_owned();
            let status = execution.status();

            if let Some(verdict) = parse_verdict_line(&stdout) {
                return Ok(SandboxVerdict {
                    passed: verdict.passed,
                    fail_reason: verdict.fail_reason,
                    stdout,
                    stderr,
                });
            }

            let fail_reason = if !stderr.trim().is_empty() {
                stderr.clone()
            } else if !stdout.trim().is_empty() {
                format!("invalid sandbox verdict: {stdout}")
            } else if !status.success {
                format!("sandbox execution failed with status {}", status.code)
            } else {
                "sandbox returned no verdict".to_string()
            };

            Ok(SandboxVerdict {
                passed: false,
                fail_reason,
                stdout,
                stderr,
            })
        }
        Err(err) => Err(format!("run python in sandbox `{name}` failed: {err}")),
    }
}

fn next_sandbox_name() -> String {
    let id = SANDBOX_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("rwkv-eval-coding-{id}")
}

fn parse_verdict_line(stdout: &str) -> Option<SandboxVerdictWire> {
    stdout
        .lines()
        .rev()
        .find_map(|line| sonic_rs::from_str::<SandboxVerdictWire>(line.trim()).ok())
}

#[cfg(test)]
mod tests {
    use super::parse_verdict_line;

    #[test]
    fn parses_last_json_line() {
        let stdout = "noise\n{\"passed\":false,\"fail_reason\":\"bad\"}\n";
        let verdict = parse_verdict_line(stdout).unwrap();
        assert!(!verdict.passed);
        assert_eq!(verdict.fail_reason, "bad");
    }
}
