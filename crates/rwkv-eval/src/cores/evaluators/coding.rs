use std::path::PathBuf;

use async_openai::{Client, config::OpenAIConfig};
use microsandbox::Sandbox;
use serde::Deserialize;
use tempfile::TempDir;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::cores::{
    datasets::SamplingConfig,
    inferers::{CompletionRequest, create_completion_streamed},
    sandbox_queue::{SandboxQueue, SandboxQueueRequest, SandboxVerdict},
};

const DEFAULT_MEMORY_MB: u32 = 512;
const DEFAULT_CPUS: u8 = 1;
const SANDBOX_WORKDIR: &str = "/work";
const SANDBOX_SCRIPT_NAME: &str = "judge.py";

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
    let resp = create_completion_streamed(model_client, &req)
        .await
        .unwrap();
    resp.choices[0].text.clone()
}

pub async fn ensure_microsandbox_available() -> Result<(), String> {
    let probe_script = r#"
import json
print(json.dumps({"passed": True, "fail_reason": ""}))
"#;

    let verdict = run_python_verdict_script_direct(probe_script)
        .await
        .map_err(|err| format!("microsandbox unavailable: {err}"))?;

    if verdict.passed {
        Ok(())
    } else {
        Err(format!(
            "microsandbox probe failed: {}",
            verdict.fail_reason
        ))
    }
}

pub async fn run_python_verdict_script(
    script: &str,
    sandbox_queue: &SandboxQueue,
) -> Result<SandboxVerdict, String> {
    let (result_tx, result_rx) = oneshot::channel();
    sandbox_queue
        .send(SandboxQueueRequest {
            script: script.to_string(),
            result_tx,
        })
        .await
        .map_err(|_| "sandbox_queue closed before request was accepted".to_string())?;
    result_rx
        .await
        .map_err(|_| "sandbox_queue closed before verdict was returned".to_string())?
}

pub(crate) async fn run_python_verdict_script_direct(
    script: &str,
) -> Result<SandboxVerdict, String> {
    let name = next_sandbox_name();
    let workdir = TempDir::new().map_err(|err| format!("create temp workdir failed: {err}"))?;
    let script_path = workdir.path().join(SANDBOX_SCRIPT_NAME);
    tokio::fs::write(&script_path, script)
        .await
        .map_err(|err| format!("write temp judge script `{}` failed: {err}", script_path.display()))?;
    let sandbox = Sandbox::builder(name.clone())
        .image("python:3.12")
        .memory(DEFAULT_MEMORY_MB)
        .cpus(DEFAULT_CPUS)
        .volume(SANDBOX_WORKDIR, |mount| mount.bind(workdir.path()))
        .create()
        .await
        .map_err(|err| format!("create sandbox `{name}` failed: {err}"))?;

    let guest_script_path = guest_script_path();
    let execution = sandbox
        .exec_with("python", |exec| {
            exec.args([guest_script_path.as_str()])
                .env("PYTHONDONTWRITEBYTECODE", "1")
        })
        .await;
    if let Err(err) = sandbox.stop_and_wait().await {
        eprintln!("failed to stop microsandbox `{name}`: {err}");
    } else if let Err(err) = sandbox.remove_persisted().await {
        eprintln!("failed to remove persisted microsandbox `{name}`: {err}");
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
    format!("rwkv-eval-coding-{}", Uuid::new_v4().simple())
}

fn guest_script_path() -> String {
    PathBuf::from(SANDBOX_WORKDIR)
        .join(SANDBOX_SCRIPT_NAME)
        .display()
        .to_string()
}

fn parse_verdict_line(stdout: &str) -> Option<SandboxVerdictWire> {
    stdout
        .lines()
        .rev()
        .find_map(|line| sonic_rs::from_str::<SandboxVerdictWire>(line.trim()).ok())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{Mutex, mpsc};

    use super::{SandboxVerdict, next_sandbox_name, parse_verdict_line, run_python_verdict_script};

    #[test]
    fn parses_last_json_line() {
        let stdout = "noise\n{\"passed\":false,\"fail_reason\":\"bad\"}\n";
        let verdict = parse_verdict_line(stdout).unwrap();
        assert!(!verdict.passed);
        assert_eq!(verdict.fail_reason, "bad");
    }

    #[test]
    fn generates_unique_sandbox_names() {
        let first = next_sandbox_name();
        let second = next_sandbox_name();
        assert!(first.starts_with("rwkv-eval-coding-"));
        assert!(second.starts_with("rwkv-eval-coding-"));
        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn sandbox_queue_processes_requests_in_submission_order() {
        let (sandbox_queue, mut sandbox_queue_rx) =
            mpsc::channel::<crate::cores::sandbox_queue::SandboxQueueRequest>(8);
        let seen_scripts = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen_scripts_consumer = Arc::clone(&seen_scripts);

        tokio::spawn(async move {
            while let Some(request) = sandbox_queue_rx.recv().await {
                seen_scripts_consumer
                    .lock()
                    .await
                    .push(request.script.clone());
                let _ = request.result_tx.send(Ok(SandboxVerdict {
                    passed: true,
                    fail_reason: String::new(),
                    stdout: request.script,
                    stderr: String::new(),
                }));
            }
        });

        let first = run_python_verdict_script("first", &sandbox_queue);
        let second = run_python_verdict_script("second", &sandbox_queue);
        let (first, second) = tokio::join!(first, second);

        assert_eq!(first.unwrap().stdout, "first");
        assert_eq!(second.unwrap().stdout, "second");
        assert_eq!(
            &*seen_scripts.lock().await,
            &["first".to_string(), "second".to_string()]
        );
    }
}
