use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{BenchError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalCommandResult {
    pub command: Vec<String>,
    pub duration_s: f64,
    pub success: bool,
    pub exit_code: Option<i32>,
}

impl ExternalCommandResult {
    pub fn command_line(&self) -> String {
        self.command.join(" ")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NsysProfileArgs {
    pub output_prefix: PathBuf,
    pub trace: String,
    pub sample: String,
    pub command: Vec<String>,
}

pub fn run_external_command(command: &[String]) -> Result<ExternalCommandResult> {
    if command.is_empty() {
        return Err(BenchError::invalid_argument("command cannot be empty"));
    }

    let mut cmd = Command::new(&command[0]);
    cmd.args(&command[1..]);

    let start = Instant::now();
    let status = cmd.status()?;
    let duration_s = start.elapsed().as_secs_f64();

    Ok(ExternalCommandResult {
        command: command.to_vec(),
        duration_s,
        success: status.success(),
        exit_code: status.code(),
    })
}

pub fn run_tracy_passthrough(command: &[String]) -> Result<ExternalCommandResult> {
    run_external_command(command)
}

pub fn run_nsys(args: &NsysProfileArgs) -> Result<ExternalCommandResult> {
    if args.command.is_empty() {
        return Err(BenchError::invalid_argument(
            "nsys target command cannot be empty",
        ));
    }

    if let Some(parent) = args.output_prefix.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut full = vec![
        "nsys".to_string(),
        "profile".to_string(),
        "--output".to_string(),
        args.output_prefix.display().to_string(),
        "--force-overwrite".to_string(),
        "true".to_string(),
        "--trace".to_string(),
        args.trace.clone(),
        "--sample".to_string(),
        args.sample.clone(),
        "--".to_string(),
    ];
    full.extend(args.command.clone());

    run_external_command(&full)
}

pub fn ensure_success(run: &ExternalCommandResult) -> Result<()> {
    if run.success {
        return Ok(());
    }

    Err(BenchError::command_failed(
        run.command_line(),
        run.exit_code,
    ))
}
