use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{SystemTime, UNIX_EPOCH},
};

use reqwest::Client;
use rwkv_config::raw::eval::RawEvalConfig;
use serde::{Deserialize, Serialize};
use sonic_rs::Value;
use tokio::{
    process::{Child, Command},
    sync::Mutex,
};

use crate::services::{ServiceError, ServiceResult};

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
    #[serde(default)]
    pub dependencies: Vec<RuntimeDependencyStatus>,
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
            dependencies: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DependencyRole {
    Model,
    Judger,
    Checker,
}

impl DependencyRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Judger => "judger",
            Self::Checker => "checker",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DependencyStatus {
    Unknown,
    Ok,
    Failed,
    Skipped,
}

impl DependencyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Ok => "ok",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeDependencyStatus {
    pub role: DependencyRole,
    pub label: String,
    pub base_url: String,
    pub status: DependencyStatus,
    pub message: Option<String>,
    pub checked_at_unix_ms: Option<u64>,
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
        let dependencies = current
            .as_ref()
            .map(|runtime| runtime.dependencies.clone())
            .unwrap_or_default();
        let runtime = RuntimeFile {
            observed_status,
            started_at_unix_ms,
            updated_at_unix_ms: now,
            finished_at_unix_ms: observed_status.is_terminal().then_some(now),
            error: error.map(ToOwned::to_owned),
            dependencies,
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

    pub fn update_dependency_status(
        &self,
        role: DependencyRole,
        label: &str,
        base_url: &str,
        status: DependencyStatus,
        message: Option<&str>,
    ) -> Result<(), String> {
        let now = current_unix_millis();
        let mut runtime = read_runtime_file(&self.runtime_path)?
            .unwrap_or_else(|| RuntimeFile::new(ObservedStatus::Starting, None));

        if let Some(existing) = runtime.dependencies.iter_mut().find(|dependency| {
            dependency.role == role && dependency.label == label && dependency.base_url == base_url
        }) {
            existing.status = status;
            existing.message = message.map(ToOwned::to_owned);
            existing.checked_at_unix_ms = Some(now);
        } else {
            runtime.dependencies.push(RuntimeDependencyStatus {
                role,
                label: label.to_string(),
                base_url: base_url.to_string(),
                status,
                message: message.map(ToOwned::to_owned),
                checked_at_unix_ms: Some(now),
            });
        }

        runtime.updated_at_unix_ms = now;
        write_runtime_file(&self.runtime_path, &runtime)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthTargetResult {
    pub base_url: String,
    pub roles: Vec<String>,
    pub status: String,
    pub error: Option<String>,
    pub health: Option<Value>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct InferHealthPayload {
    status: String,
    window_seconds: u64,
    server_time_unix_ms: u64,
    gpu_panels: Vec<Value>,
}

pub async fn fetch_health_targets(
    client: &Client,
    cfg: &RawEvalConfig,
) -> ServiceResult<Vec<HealthTargetResult>> {
    fetch_health_targets_for_targets(client, collect_health_targets(cfg)).await
}

pub async fn fetch_admin_health_targets(
    client: &Client,
    snapshot: Option<&EvalRunSnapshot>,
    service_cfg: &RawEvalConfig,
) -> ServiceResult<Vec<HealthTargetResult>> {
    let targets = snapshot
        .map(|snapshot| snapshot.health_targets.clone())
        .unwrap_or_else(|| collect_health_targets(service_cfg));
    fetch_health_targets_for_targets(client, targets).await
}

async fn fetch_health_targets_for_targets(
    client: &Client,
    targets: Vec<HealthTarget>,
) -> ServiceResult<Vec<HealthTargetResult>> {
    let mut results = Vec::with_capacity(targets.len());

    for target in targets {
        let url = normalize_base_url(&target.base_url);
        let endpoint = format!("{}/health", url.trim_end_matches('/'));
        match client.get(&endpoint).send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    results.push(HealthTargetResult {
                        base_url: target.base_url,
                        roles: target.roles,
                        status: "unreachable".to_string(),
                        error: Some(format!("GET {endpoint} returned HTTP {status}")),
                        health: None,
                    });
                    continue;
                }

                let body = match response.bytes().await {
                    Ok(body) => body,
                    Err(err) => {
                        results.push(HealthTargetResult {
                            base_url: target.base_url,
                            roles: target.roles,
                            status: "invalid_response".to_string(),
                            error: Some(format!("read {endpoint} failed: {err}")),
                            health: None,
                        });
                        continue;
                    }
                };

                let health = match sonic_rs::from_slice::<Value>(body.as_ref()) {
                    Ok(health) => health,
                    Err(err) => {
                        results.push(HealthTargetResult {
                            base_url: target.base_url,
                            roles: target.roles,
                            status: "invalid_response".to_string(),
                            error: Some(format!("decode {endpoint} failed: {err}")),
                            health: None,
                        });
                        continue;
                    }
                };
                if sonic_rs::from_slice::<InferHealthPayload>(body.as_ref()).is_err() {
                    results.push(HealthTargetResult {
                        base_url: target.base_url,
                        roles: target.roles,
                        status: "invalid_response".to_string(),
                        error: Some(format!(
                            "GET {endpoint} returned a non-rwkv-infer health payload"
                        )),
                        health: None,
                    });
                    continue;
                }
                results.push(HealthTargetResult {
                    base_url: target.base_url,
                    roles: target.roles,
                    status: "ok".to_string(),
                    error: None,
                    health: Some(health),
                });
            }
            Err(err) => results.push(HealthTargetResult {
                base_url: target.base_url,
                roles: target.roles,
                status: "unreachable".to_string(),
                error: Some(format!("GET {endpoint} failed: {err}")),
                health: None,
            }),
        }
    }

    Ok(results)
}

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
    health_targets: Vec<HealthTarget>,
    child: Child,
}

#[derive(Clone, Debug)]
pub struct EvalRunSnapshot {
    pub config_path: String,
    pub desired_state: DesiredState,
    pub runtime: RuntimeFile,
    health_targets: Vec<HealthTarget>,
}

impl EvalController {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(EvalControllerState { active: None }),
        }
    }

    pub async fn start(&self, run_cfg: &RawEvalConfig) -> ServiceResult<EvalRunSnapshot> {
        let mut guard = self.inner.lock().await;
        refresh_active_run(guard.active.as_mut())?;

        if let Some(active) = guard.active.as_mut() {
            if !active.snapshot()?.runtime.observed_status.is_terminal() {
                return Err(ServiceError::conflict(
                    "an evaluation task is already active",
                ));
            }
        }

        let active = ActiveEvalRun::spawn(run_cfg)?;
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
    fn spawn(run_cfg: &RawEvalConfig) -> ServiceResult<Self> {
        let mut run_cfg = run_cfg.clone();
        run_cfg.fill_default();
        run_cfg.admin_api_key = None;
        run_cfg.upload_to_space = Some(true);

        if run_cfg.run_mode.as_deref() == Some("resume") {
            return Err(ServiceError::bad_request(
                "admin start does not support run_mode=resume with temporary config paths",
            ));
        }

        let temp_dir = build_temp_dir();
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
        let mut runtime = RuntimeFile::new(ObservedStatus::Starting, None);
        runtime.dependencies = build_runtime_dependencies(&run_cfg);
        write_runtime_file(&runtime_path, &runtime).map_err(ServiceError::internal)?;
        let health_targets = collect_health_targets(&run_cfg);

        let mut command = build_evaluator_command(&temp_dir)?;
        command
            .env(CONTROL_PATH_ENV, &control_path)
            .env(RUNTIME_PATH_ENV, &runtime_path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        let child = command
            .spawn()
            .map_err(|err| ServiceError::internal(format!("spawn evaluator failed: {err}")))?;

        Ok(Self {
            config_path: config_path.display().to_string(),
            control_path,
            runtime_path,
            health_targets,
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
            health_targets: self.health_targets.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HealthTarget {
    base_url: String,
    roles: Vec<String>,
}

fn collect_health_targets(cfg: &RawEvalConfig) -> Vec<HealthTarget> {
    let mut dedup = BTreeMap::<String, BTreeSet<String>>::new();

    for model in &cfg.models {
        dedup
            .entry(model.base_url.clone())
            .or_default()
            .insert(format!("model:{}", model.model));
    }

    dedup
        .into_iter()
        .map(|(base_url, roles)| HealthTarget {
            base_url,
            roles: roles.into_iter().collect(),
        })
        .collect()
}

fn normalize_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

pub fn build_runtime_dependencies(cfg: &RawEvalConfig) -> Vec<RuntimeDependencyStatus> {
    let mut dependencies = cfg
        .models
        .iter()
        .map(|model| RuntimeDependencyStatus {
            role: DependencyRole::Model,
            label: model.model.clone(),
            base_url: model.base_url.clone(),
            status: DependencyStatus::Unknown,
            message: None,
            checked_at_unix_ms: None,
        })
        .collect::<Vec<_>>();

    dependencies.push(RuntimeDependencyStatus {
        role: DependencyRole::Judger,
        label: cfg.llm_judger.model.clone(),
        base_url: cfg.llm_judger.base_url.clone(),
        status: DependencyStatus::Unknown,
        message: None,
        checked_at_unix_ms: None,
    });
    dependencies.push(RuntimeDependencyStatus {
        role: DependencyRole::Checker,
        label: cfg.llm_checker.model.clone(),
        base_url: cfg.llm_checker.base_url.clone(),
        status: if cfg.skip_checker.unwrap_or(false) {
            DependencyStatus::Skipped
        } else {
            DependencyStatus::Unknown
        },
        message: None,
        checked_at_unix_ms: None,
    });

    dependencies
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
            dependencies: runtime.dependencies,
        },
    )
    .map_err(ServiceError::internal)
}

fn build_temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "rwkv-lm-eval-api-{}-{}",
        std::process::id(),
        current_unix_millis()
    ))
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

fn read_control_file(path: &Path) -> Result<Option<ControlFile>, String> {
    read_json(path)
}

fn write_control_file(path: &Path, file: &ControlFile) -> Result<(), String> {
    write_json(path, file)
}

fn read_runtime_file(path: &Path) -> Result<Option<RuntimeFile>, String> {
    read_json(path)
}

fn write_runtime_file(path: &Path, file: &RuntimeFile) -> Result<(), String> {
    write_json(path, file)
}

fn current_unix_millis() -> u64 {
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

    use axum::{Router, http::header::CONTENT_TYPE, routing::get};
    use reqwest::Client;
    use rwkv_config::raw::eval::{ExtApiConfig, IntApiConfig, RawEvalConfig, SpaceDbConfig};
    use tokio::{net::TcpListener, task::JoinHandle};

    use super::{
        ControlFile,
        DependencyRole,
        DependencyStatus,
        DesiredState,
        EvalRunSnapshot,
        EvalRuntimeControl,
        ObservedStatus,
        RuntimeDependencyStatus,
        RuntimeFile,
        build_runtime_dependencies,
        collect_health_targets,
        current_unix_millis,
        fetch_admin_health_targets,
        fetch_health_targets,
        normalize_base_url,
        read_control_file,
        read_runtime_file,
        workspace_root,
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

    fn model_config(base_url: impl Into<String>, model: &str) -> IntApiConfig {
        IntApiConfig {
            model_arch_version: "rwkv7".to_string(),
            model_data_version: "g1".to_string(),
            model_num_params: "1.5b".to_string(),
            base_url: base_url.into(),
            api_key: "secret".to_string(),
            model: model.to_string(),
            max_batch_size: Some(1),
        }
    }

    async fn spawn_server(router: Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{addr}"), handle)
    }

    fn sample_cfg() -> RawEvalConfig {
        RawEvalConfig {
            experiment_name: "demo".to_string(),
            experiment_desc: "demo".to_string(),
            admin_api_key: None,
            run_mode: Some("new".to_string()),
            skip_checker: Some(false),
            skip_dataset_check: Some(false),
            judger_concurrency: Some(1),
            checker_concurrency: Some(1),
            db_pool_max_connections: Some(1),
            model_arch_versions: vec!["rwkv7".to_string()],
            model_data_versions: vec!["g1".to_string()],
            model_num_params: vec!["1.5b".to_string()],
            benchmark_field: vec!["Knowledge".to_string()],
            extra_benchmark_name: vec![],
            upload_to_space: Some(true),
            git_hash: "abc".to_string(),
            models: vec![
                model_config("127.0.0.1:8080", "demo-a"),
                model_config("127.0.0.1:8080", "demo-b"),
            ],
            llm_judger: ExtApiConfig {
                base_url: "judge:8080".to_string(),
                api_key: "secret".to_string(),
                model: "judge".to_string(),
            },
            llm_checker: ExtApiConfig {
                base_url: "judge:8080".to_string(),
                api_key: "secret".to_string(),
                model: "checker".to_string(),
            },
            space_db: SpaceDbConfig::default(),
        }
    }

    #[test]
    fn collects_unique_model_base_urls() {
        let targets = collect_health_targets(&sample_cfg());
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].roles,
            vec!["model:demo-a".to_string(), "model:demo-b".to_string()]
        );
    }

    #[test]
    fn builds_dependency_statuses_and_marks_skipped_checker() {
        let mut cfg = sample_cfg();
        cfg.skip_checker = Some(true);

        let dependencies = build_runtime_dependencies(&cfg);

        assert_eq!(dependencies.len(), 4);
        assert!(dependencies.iter().any(|dependency| {
            dependency.role == DependencyRole::Judger
                && dependency.status == DependencyStatus::Unknown
        }));
        assert!(dependencies.iter().any(|dependency| {
            dependency.role == DependencyRole::Checker
                && dependency.status == DependencyStatus::Skipped
                && dependency.checked_at_unix_ms.is_none()
        }));
    }

    #[test]
    fn normalizes_scheme() {
        assert_eq!(
            normalize_base_url("127.0.0.1:8080"),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            normalize_base_url("https://example.com"),
            "https://example.com"
        );
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
        let runtime = read_runtime_file(&path).unwrap().unwrap();
        assert_eq!(runtime.finished_at_unix_ms, None);

        write_runtime_file(&path, &RuntimeFile::new(ObservedStatus::Completed, None)).unwrap();
        let runtime = read_runtime_file(&path).unwrap().unwrap();
        assert!(runtime.finished_at_unix_ms.is_some());
    }

    #[test]
    fn write_status_preserves_dependency_statuses() {
        let control_path = temp_path("control.json");
        let runtime_path = temp_path("runtime.json");
        let control = EvalRuntimeControl {
            control_path,
            runtime_path: runtime_path.clone(),
        };
        let dependency = RuntimeDependencyStatus {
            role: DependencyRole::Model,
            label: "demo-a".to_string(),
            base_url: "127.0.0.1:8080".to_string(),
            status: DependencyStatus::Ok,
            message: None,
            checked_at_unix_ms: Some(123),
        };
        let mut runtime = RuntimeFile::new(ObservedStatus::Starting, None);
        runtime.dependencies = vec![dependency.clone()];
        write_runtime_file(&runtime_path, &runtime).unwrap();

        control.write_status(ObservedStatus::Running, None).unwrap();

        let runtime = read_runtime_file(&runtime_path).unwrap().unwrap();
        assert_eq!(runtime.dependencies, vec![dependency]);
        assert_eq!(runtime.observed_status, ObservedStatus::Running);
    }

    #[tokio::test]
    async fn fetch_health_targets_only_returns_infer_targets_and_keeps_partial_failures() {
        let (ok_base_url, ok_handle) = spawn_server(
            Router::new().route(
                "/health",
                get(|| async {
                    (
                        [(CONTENT_TYPE, "application/json")],
                        r#"{"status":"ok","window_seconds":60,"server_time_unix_ms":1,"gpu_panels":[]}"#,
                    )
                }),
            ),
        )
        .await;
        let (invalid_json_base_url, invalid_json_handle) =
            spawn_server(Router::new().route("/health", get(|| async { "not-json" }))).await;
        let (wrong_schema_base_url, wrong_schema_handle) = spawn_server(Router::new().route(
            "/health",
            get(|| async { ([(CONTENT_TYPE, "application/json")], r#"{"ok":true}"#) }),
        ))
        .await;

        let mut cfg = sample_cfg();
        cfg.models = vec![
            model_config(ok_base_url.clone(), "demo-a"),
            model_config(invalid_json_base_url.clone(), "demo-b"),
            model_config(wrong_schema_base_url.clone(), "demo-c"),
        ];

        let results = fetch_health_targets(&Client::new(), &cfg).await.unwrap();

        ok_handle.abort();
        invalid_json_handle.abort();
        wrong_schema_handle.abort();

        assert_eq!(results.len(), 3);

        let ok_target = results
            .iter()
            .find(|target| target.base_url == ok_base_url)
            .unwrap();
        assert_eq!(ok_target.status, "ok");
        assert!(ok_target.health.is_some());

        let invalid_json_target = results
            .iter()
            .find(|target| target.base_url == invalid_json_base_url)
            .unwrap();
        assert_eq!(invalid_json_target.status, "invalid_response");
        assert!(
            invalid_json_target
                .error
                .as_ref()
                .is_some_and(|error| error.contains("decode"))
        );

        let wrong_schema_target = results
            .iter()
            .find(|target| target.base_url == wrong_schema_base_url)
            .unwrap();
        assert_eq!(wrong_schema_target.status, "invalid_response");
        assert!(
            wrong_schema_target
                .error
                .as_ref()
                .is_some_and(|error| { error.contains("non-rwkv-infer health payload") })
        );
    }

    #[tokio::test]
    async fn fetch_admin_health_targets_prefers_active_run_targets() {
        let (service_base_url, service_handle) = spawn_server(Router::new().route(
            "/health",
            get(|| async { (axum::http::StatusCode::BAD_GATEWAY, "service-config") }),
        ))
        .await;
        let (active_base_url, active_handle) = spawn_server(
            Router::new().route(
                "/health",
                get(|| async {
                    (
                        [(CONTENT_TYPE, "application/json")],
                        r#"{"status":"ok","window_seconds":60,"server_time_unix_ms":1,"gpu_panels":[]}"#,
                    )
                }),
            ),
        )
        .await;

        let mut service_cfg = sample_cfg();
        service_cfg.models = vec![model_config(service_base_url.clone(), "service-model")];
        let snapshot = EvalRunSnapshot {
            config_path: "/tmp/active.toml".to_string(),
            desired_state: DesiredState::Running,
            runtime: RuntimeFile::new(ObservedStatus::Running, None),
            health_targets: collect_health_targets(&RawEvalConfig {
                models: vec![
                    model_config(active_base_url.clone(), "active-a"),
                    model_config(active_base_url.clone(), "active-b"),
                ],
                ..sample_cfg()
            }),
        };

        let results = fetch_admin_health_targets(&Client::new(), Some(&snapshot), &service_cfg)
            .await
            .unwrap();

        service_handle.abort();
        active_handle.abort();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].base_url, active_base_url);
        assert_eq!(results[0].status, "ok");
        assert_eq!(
            results[0].roles,
            vec!["model:active-a".to_string(), "model:active-b".to_string()]
        );
    }

    #[tokio::test]
    async fn fetch_admin_health_targets_falls_back_to_service_config_without_active_run() {
        let (service_base_url, service_handle) = spawn_server(
            Router::new().route(
                "/health",
                get(|| async {
                    (
                        [(CONTENT_TYPE, "application/json")],
                        r#"{"status":"ok","window_seconds":60,"server_time_unix_ms":1,"gpu_panels":[]}"#,
                    )
                }),
            ),
        )
        .await;

        let mut service_cfg = sample_cfg();
        service_cfg.models = vec![model_config(service_base_url.clone(), "service-model")];

        let results = fetch_admin_health_targets(&Client::new(), None, &service_cfg)
            .await
            .unwrap();

        service_handle.abort();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].base_url, service_base_url);
        assert_eq!(results[0].status, "ok");
        assert_eq!(results[0].roles, vec!["model:service-model".to_string()]);
    }

    #[test]
    fn runtime_control_requires_both_env_paths() {
        unsafe {
            std::env::remove_var("RWKV_LM_EVAL_CONTROL_PATH");
            std::env::remove_var("RWKV_LM_EVAL_RUNTIME_PATH");
            std::env::set_var("RWKV_LM_EVAL_CONTROL_PATH", "control.json");
        }
        let err = EvalRuntimeControl::from_env().unwrap_err();
        assert!(err.contains("must be set together"));
        unsafe {
            std::env::remove_var("RWKV_LM_EVAL_CONTROL_PATH");
        }
    }

    #[test]
    fn resolves_workspace_root() {
        assert!(workspace_root().join("Cargo.toml").is_file());
    }
}
