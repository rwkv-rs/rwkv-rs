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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthTargetResult {
    pub base_url: String,
    pub roles: Vec<String>,
    pub status: String,
    pub error: Option<String>,
    pub health: Option<Value>,
}

pub async fn fetch_health_targets(
    client: &Client,
    cfg: &RawEvalConfig,
) -> ServiceResult<Vec<HealthTargetResult>> {
    let targets = collect_health_targets(cfg);
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

                let health = response.json::<Value>().await.map_err(|err| {
                    ServiceError::internal(format!("decode {endpoint} failed: {err}"))
                })?;
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
        write_runtime_file(
            &runtime_path,
            &RuntimeFile::new(ObservedStatus::Starting, None),
        )
        .map_err(ServiceError::internal)?;

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
        .entry(cfg.llm_judger.base_url.clone())
        .or_default()
        .insert(format!("judger:{}", cfg.llm_judger.model));
    dedup
        .entry(cfg.llm_checker.base_url.clone())
        .or_default()
        .insert(format!("checker:{}", cfg.llm_checker.model));

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

    use rwkv_config::raw::eval::{ExtApiConfig, IntApiConfig, RawEvalConfig, SpaceDbConfig};

    use super::{
        ControlFile,
        DesiredState,
        EvalRuntimeControl,
        ObservedStatus,
        RuntimeFile,
        collect_health_targets,
        current_unix_millis,
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

    fn sample_cfg() -> RawEvalConfig {
        RawEvalConfig {
            experiment_name: "demo".to_string(),
            experiment_desc: "demo".to_string(),
            admin_api_key: None,
            run_mode: Some("new".to_string()),
            skip_checker: Some(false),
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
                IntApiConfig {
                    model_arch_version: "rwkv7".to_string(),
                    model_data_version: "g1".to_string(),
                    model_num_params: "1.5b".to_string(),
                    base_url: "127.0.0.1:8080".to_string(),
                    api_key: "secret".to_string(),
                    model: "demo-a".to_string(),
                    max_batch_size: Some(1),
                },
                IntApiConfig {
                    model_arch_version: "rwkv7".to_string(),
                    model_data_version: "g1".to_string(),
                    model_num_params: "3b".to_string(),
                    base_url: "127.0.0.1:8080".to_string(),
                    api_key: "secret".to_string(),
                    model: "demo-b".to_string(),
                    max_batch_size: Some(1),
                },
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
    fn collects_unique_base_urls() {
        let targets = collect_health_targets(&sample_cfg());
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().any(|target| target.roles.len() == 2));
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
