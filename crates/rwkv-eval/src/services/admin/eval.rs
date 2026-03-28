use std::{
    fs,
    path::{Path, PathBuf},
    process::Stdio,
};

use rwkv_config::raw::eval::RawEvalConfig;
use tokio::{
    process::{Child, Command},
    sync::Mutex,
};

use crate::services::{
    ServiceError,
    ServiceResult,
    runtime::{
        ControlFile,
        DesiredState,
        ObservedStatus,
        RuntimeFile,
        current_unix_millis,
        read_control_file,
        read_runtime_file,
        write_control_file,
        write_runtime_file,
    },
};

pub struct EvalController {
    inner: Mutex<EvalControllerState>,
}

struct EvalControllerState {
    active: Option<ActiveEvalRun>,
}

struct ActiveEvalRun {
    config_path: String,
    control_path: PathBuf,
    runtime_path: PathBuf,
    child: Child,
}

#[derive(Clone, Debug)]
pub struct EvalRunSnapshot {
    pub config_path: String,
    pub desired_state: DesiredState,
    pub runtime: RuntimeFile,
}

impl EvalController {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(EvalControllerState { active: None }),
        }
    }

    pub async fn start(&self, service_cfg: &RawEvalConfig) -> ServiceResult<EvalRunSnapshot> {
        let mut guard = self.inner.lock().await;
        refresh_active_run(guard.active.as_mut())?;

        if let Some(active) = guard.active.as_mut() {
            if !active.snapshot()?.runtime.observed_status.is_terminal() {
                return Err(ServiceError::conflict(
                    "an evaluation task is already active",
                ));
            }
        }

        let active = ActiveEvalRun::spawn(service_cfg)?;
        let snapshot = active.snapshot()?;
        guard.active = Some(active);
        Ok(snapshot)
    }

    pub async fn pause(&self) -> ServiceResult<EvalRunSnapshot> {
        self.set_desired_state(DesiredState::Paused).await
    }

    pub async fn resume(&self) -> ServiceResult<EvalRunSnapshot> {
        self.set_desired_state(DesiredState::Running).await
    }

    pub async fn cancel(&self) -> ServiceResult<EvalRunSnapshot> {
        self.set_desired_state(DesiredState::Cancelled).await
    }

    pub async fn snapshot(&self) -> ServiceResult<Option<EvalRunSnapshot>> {
        let mut guard = self.inner.lock().await;
        refresh_active_run(guard.active.as_mut())?;

        guard
            .active
            .as_ref()
            .map(ActiveEvalRun::snapshot)
            .transpose()
    }

    async fn set_desired_state(
        &self,
        desired_state: DesiredState,
    ) -> ServiceResult<EvalRunSnapshot> {
        let mut guard = self.inner.lock().await;
        refresh_active_run(guard.active.as_mut())?;
        let active = guard
            .active
            .as_ref()
            .ok_or_else(|| ServiceError::not_found("no evaluation task has been started"))?;
        let snapshot = active.snapshot()?;
        if snapshot.runtime.observed_status.is_terminal() {
            return Err(ServiceError::conflict(format!(
                "evaluation is already {}",
                snapshot.runtime.observed_status.as_str()
            )));
        }

        write_control_file(
            &active.control_path,
            &ControlFile {
                desired_state,
                updated_at_unix_ms: current_unix_millis(),
            },
        )
        .map_err(ServiceError::internal)?;

        active.snapshot()
    }
}

impl ActiveEvalRun {
    fn spawn(service_cfg: &RawEvalConfig) -> ServiceResult<Self> {
        let mut run_cfg = service_cfg.clone();
        run_cfg.fill_default();
        run_cfg.admin_api_key = None;
        run_cfg.upload_to_space = Some(true);

        if run_cfg.run_mode.as_deref() == Some("resume") {
            return Err(ServiceError::bad_request(
                "admin start does not support run_mode=resume with temporary config paths",
            ));
        }

        let temp_dir = build_temp_dir()?;
        fs::create_dir_all(&temp_dir).map_err(|err| {
            ServiceError::internal(format!(
                "create temp eval dir {} failed: {err}",
                temp_dir.display()
            ))
        })?;
        let config_path = temp_dir.join("active.toml");
        let control_path = temp_dir.join("control.json");
        let runtime_path = temp_dir.join("runtime.json");

        let raw = toml::to_string_pretty(&run_cfg).map_err(|err| {
            ServiceError::internal(format!("serialize eval config failed: {err}"))
        })?;
        fs::write(&config_path, raw).map_err(|err| {
            ServiceError::internal(format!("write {} failed: {err}", config_path.display()))
        })?;

        write_control_file(
            &control_path,
            &ControlFile {
                desired_state: DesiredState::Running,
                updated_at_unix_ms: current_unix_millis(),
            },
        )
        .map_err(ServiceError::internal)?;
        write_runtime_file(
            &runtime_path,
            &RuntimeFile::new(ObservedStatus::Starting, None),
        )
        .map_err(ServiceError::internal)?;

        let mut command = build_evaluator_command(&temp_dir)?;
        command
            .env("RWKV_LM_EVAL_CONTROL_PATH", &control_path)
            .env("RWKV_LM_EVAL_RUNTIME_PATH", &runtime_path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        let child = command
            .spawn()
            .map_err(|err| ServiceError::internal(format!("spawn evaluator failed: {err}")))?;

        Ok(Self {
            config_path: config_path.display().to_string(),
            control_path,
            runtime_path,
            child,
        })
    }

    fn snapshot(&self) -> ServiceResult<EvalRunSnapshot> {
        let desired_state = read_control_file(&self.control_path)
            .map_err(ServiceError::internal)?
            .map(|control| control.desired_state)
            .unwrap_or(DesiredState::Running);
        let runtime = read_runtime_file(&self.runtime_path)
            .map_err(ServiceError::internal)?
            .unwrap_or_else(|| RuntimeFile::new(ObservedStatus::Starting, None));

        Ok(EvalRunSnapshot {
            config_path: self.config_path.clone(),
            desired_state,
            runtime,
        })
    }
}

fn refresh_active_run(active: Option<&mut ActiveEvalRun>) -> ServiceResult<()> {
    let Some(active) = active else {
        return Ok(());
    };

    let Some(status) = active
        .child
        .try_wait()
        .map_err(|err| ServiceError::internal(format!("poll evaluator process failed: {err}")))?
    else {
        return Ok(());
    };

    let runtime = read_runtime_file(&active.runtime_path)
        .map_err(ServiceError::internal)?
        .unwrap_or_else(|| RuntimeFile::new(ObservedStatus::Starting, None));
    if runtime.observed_status.is_terminal() {
        return Ok(());
    }

    let next_status = if status.success() {
        ObservedStatus::Completed
    } else {
        ObservedStatus::Failed
    };
    let error = (!status.success()).then(|| format!("evaluator exited with status {status}"));

    write_runtime_file(
        &active.runtime_path,
        &RuntimeFile {
            observed_status: next_status,
            started_at_unix_ms: runtime.started_at_unix_ms,
            updated_at_unix_ms: current_unix_millis(),
            finished_at_unix_ms: Some(current_unix_millis()),
            error,
        },
    )
    .map_err(ServiceError::internal)
}

fn build_temp_dir() -> ServiceResult<PathBuf> {
    let dir = std::env::temp_dir().join(format!(
        "rwkv-lm-eval-api-{}-{}",
        std::process::id(),
        current_unix_millis()
    ));
    Ok(dir)
}

fn build_evaluator_command(config_dir: &Path) -> ServiceResult<Command> {
    if let Some(path) = compiled_example_path()? {
        let mut command = Command::new(path);
        command
            .arg("--config-dir")
            .arg(config_dir)
            .arg("--eval-config")
            .arg("active");
        return Ok(command);
    }

    let mut command = Command::new("cargo");
    command
        .current_dir(workspace_root())
        .arg("run")
        .arg("-p")
        .arg("rwkv-lm-eval")
        .arg("--example")
        .arg("rwkv-lm-eval-test")
        .arg("--")
        .arg("--config-dir")
        .arg(config_dir)
        .arg("--eval-config")
        .arg("active");
    Ok(command)
}

fn compiled_example_path() -> ServiceResult<Option<PathBuf>> {
    let current_exe = std::env::current_exe().map_err(|err| {
        ServiceError::internal(format!("resolve current executable failed: {err}"))
    })?;
    let suffix = std::env::consts::EXE_SUFFIX;
    let sibling = current_exe.with_file_name(format!("rwkv-lm-eval-test{suffix}"));
    Ok(sibling.is_file().then_some(sibling))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[cfg(test)]
mod tests {
    use super::workspace_root;

    #[test]
    fn resolves_workspace_root() {
        assert!(workspace_root().join("Cargo.toml").is_file());
    }
}
