use std::{process::Command, time::Instant};

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

pub fn ensure_success(run: &ExternalCommandResult) -> Result<()> {
    if run.success {
        return Ok(());
    }

    Err(BenchError::command_failed(
        run.command_line(),
        run.exit_code,
    ))
}
