use std::{
    env,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

const CONTROL_PATH_ENV: &str = "RWKV_LM_EVAL_CONTROL_PATH";
const RUNTIME_PATH_ENV: &str = "RWKV_LM_EVAL_RUNTIME_PATH";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DesiredState {
    Running,
    Paused,
    Cancelled,
}

impl DesiredState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservedStatus {
    Starting,
    Running,
    Pausing,
    Paused,
    Cancelling,
    Cancelled,
    Completed,
    Failed,
}

impl ObservedStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Pausing => "pausing",
            Self::Paused => "paused",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Completed | Self::Failed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControlFile {
    pub desired_state: DesiredState,
    pub updated_at_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeFile {
    pub observed_status: ObservedStatus,
    pub started_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
    pub finished_at_unix_ms: Option<u64>,
    pub error: Option<String>,
}

impl RuntimeFile {
    pub fn new(observed_status: ObservedStatus, error: Option<String>) -> Self {
        let now = current_unix_millis();
        Self {
            observed_status,
            started_at_unix_ms: now,
            updated_at_unix_ms: now,
            finished_at_unix_ms: observed_status.is_terminal().then_some(now),
            error,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalRuntimeControl {
    control_path: PathBuf,
    runtime_path: PathBuf,
}

impl EvalRuntimeControl {
    pub fn from_env() -> Result<Option<Self>, String> {
        let control_path = env::var_os(CONTROL_PATH_ENV);
        let runtime_path = env::var_os(RUNTIME_PATH_ENV);

        match (control_path, runtime_path) {
            (None, None) => Ok(None),
            (Some(control_path), Some(runtime_path)) => Ok(Some(Self {
                control_path: PathBuf::from(control_path),
                runtime_path: PathBuf::from(runtime_path),
            })),
            _ => Err(format!(
                "{CONTROL_PATH_ENV} and {RUNTIME_PATH_ENV} must be set together"
            )),
        }
    }

    pub fn desired_state(&self) -> Result<DesiredState, String> {
        Ok(read_control_file(&self.control_path)?
            .map(|file| file.desired_state)
            .unwrap_or(DesiredState::Running))
    }

    pub fn write_status(
        &self,
        observed_status: ObservedStatus,
        error: Option<&str>,
    ) -> Result<(), String> {
        let now = current_unix_millis();
        let current = read_runtime_file(&self.runtime_path)?;
        let started_at_unix_ms = current
            .as_ref()
            .map(|runtime| runtime.started_at_unix_ms)
            .unwrap_or(now);
        let runtime = RuntimeFile {
            observed_status,
            started_at_unix_ms,
            updated_at_unix_ms: now,
            finished_at_unix_ms: observed_status.is_terminal().then_some(now),
            error: error.map(ToOwned::to_owned),
        };
        write_runtime_file(&self.runtime_path, &runtime)
    }

    pub fn heartbeat(&self) -> Result<(), String> {
        let Some(mut runtime) = read_runtime_file(&self.runtime_path)? else {
            return Ok(());
        };
        runtime.updated_at_unix_ms = current_unix_millis();
        write_runtime_file(&self.runtime_path, &runtime)
    }
}

pub fn read_control_file(path: &Path) -> Result<Option<ControlFile>, String> {
    read_json(path)
}

pub fn write_control_file(path: &Path, file: &ControlFile) -> Result<(), String> {
    write_json(path, file)
}

pub fn read_runtime_file(path: &Path) -> Result<Option<RuntimeFile>, String> {
    read_json(path)
}

pub fn write_runtime_file(path: &Path, file: &RuntimeFile) -> Result<(), String> {
    write_json(path, file)
}

pub fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn read_json<T>(path: &Path) -> Result<Option<T>, String>
where
    T: for<'de> Deserialize<'de>,
{
    if !path.is_file() {
        return Ok(None);
    }

    let raw =
        fs::read_to_string(path).map_err(|err| format!("read {} failed: {err}", path.display()))?;
    sonic_rs::from_str(&raw).map(Some).map_err(|err| {
        format!(
            "parse runtime control json {} failed: {err}",
            path.display()
        )
    })
}

fn write_json<T>(path: &Path, value: &T) -> Result<(), String>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("create dir {} failed: {err}", parent.display()))?;
    }

    let tmp_path = path.with_extension("tmp");
    let raw = sonic_rs::to_string(value)
        .map_err(|err| format!("serialize {} failed: {err}", path.display()))?;
    fs::write(&tmp_path, raw)
        .map_err(|err| format!("write {} failed: {err}", tmp_path.display()))?;
    fs::rename(&tmp_path, path).map_err(|err| {
        format!(
            "replace runtime control {} from {} failed: {err}",
            path.display(),
            tmp_path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{
        ControlFile,
        DesiredState,
        EvalRuntimeControl,
        ObservedStatus,
        RuntimeFile,
        current_unix_millis,
        read_control_file,
        read_runtime_file,
        write_control_file,
        write_runtime_file,
    };

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rwkv-lm-eval-runtime-control-{}-{}",
            std::process::id(),
            current_unix_millis()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn reads_and_writes_control_files() {
        let path = temp_path("control.json");
        write_control_file(
            &path,
            &ControlFile {
                desired_state: DesiredState::Paused,
                updated_at_unix_ms: 1,
            },
        )
        .unwrap();

        let file = read_control_file(&path).unwrap().unwrap();
        assert_eq!(file.desired_state, DesiredState::Paused);
    }

    #[test]
    fn writes_terminal_runtime_timestamps() {
        let path = temp_path("runtime.json");
        write_runtime_file(&path, &RuntimeFile::new(ObservedStatus::Starting, None)).unwrap();

        let control = EvalRuntimeControl {
            control_path: temp_path("unused-control.json"),
            runtime_path: path.clone(),
        };
        control
            .write_status(ObservedStatus::Completed, Some("done"))
            .unwrap();

        let runtime = read_runtime_file(&path).unwrap().unwrap();
        assert_eq!(runtime.observed_status, ObservedStatus::Completed);
        assert!(runtime.finished_at_unix_ms.is_some());
        assert_eq!(runtime.error.as_deref(), Some("done"));
    }
}
