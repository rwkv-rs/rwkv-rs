use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use rwkv_config::raw::infer::{GenerationConfig, RawInferConfig, RawIpcConfig};
use rwkv_config::raw::model::RawModelConfig;
use rwkv_config::validated::model::{FinalModelConfig, FinalModelConfigBuilder};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::service::{ModelRuntimeGroup, Service};

fn is_path_like(s: &str) -> bool {
    s.contains('/') || s.contains('\\')
}

fn resolve_path(base_dir: &Path, path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        base_dir.join(p).to_string_lossy().to_string()
    }
}

fn resolve_model_cfg_path(config_dir: &Path, infer_cfg_dir: &Path, model_cfg: &str) -> PathBuf {
    if is_path_like(model_cfg) {
        let p = PathBuf::from(model_cfg);
        if p.is_absolute() {
            p
        } else {
            infer_cfg_dir.join(p)
        }
    } else {
        config_dir.join("model").join(format!("{model_cfg}.toml"))
    }
}

fn validate_generation_models(models: &[GenerationConfig]) -> crate::Result<()> {
    if models.is_empty() {
        return Err(crate::Error::bad_request(
            "infer config requires at least one model",
        ));
    }

    let mut names = HashSet::new();
    for model in models {
        if model.model_name.trim().is_empty() {
            return Err(crate::Error::bad_request("model_name cannot be empty"));
        }
        if !names.insert(model.model_name.clone()) {
            return Err(crate::Error::bad_request(format!(
                "duplicated model_name: {}",
                model.model_name
            )));
        }
        if model.model_cfg.trim().is_empty() {
            return Err(crate::Error::bad_request(format!(
                "model_cfg cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.weights_path.trim().is_empty() {
            return Err(crate::Error::bad_request(format!(
                "weights_path cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.tokenizer_vocab_path.trim().is_empty() {
            return Err(crate::Error::bad_request(format!(
                "tokenizer_vocab_path cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.device_ids.is_empty() {
            return Err(crate::Error::bad_request(format!(
                "device_ids cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.max_batch_size.unwrap_or_default() < 1 {
            return Err(crate::Error::bad_request(format!(
                "max_batch_size must be >= 1 for model {}",
                model.model_name
            )));
        }
        if model.max_context_len.unwrap_or_default() < 1 {
            return Err(crate::Error::bad_request(format!(
                "max_context_len must be >= 1 for model {}",
                model.model_name
            )));
        }
    }

    Ok(())
}

fn read_toml_file<T: for<'de> Deserialize<'de>>(path: &Path) -> crate::Result<T> {
    let content = fs::read_to_string(path).map_err(|e| {
        crate::Error::bad_request_with_source(format!("failed to read {}", path.display()), e)
    })?;
    toml::from_str(&content).map_err(|e| {
        crate::Error::bad_request_with_source(format!("invalid toml {}", path.display()), e)
    })
}

fn load_raw_infer_cfg(path: &Path) -> crate::Result<RawInferConfig> {
    let mut cfg: RawInferConfig = read_toml_file(path)?;
    cfg.fill_default();
    validate_generation_models(&cfg.models)?;
    Ok(cfg)
}

fn load_model_cfgs(
    config_dir: &Path,
    infer_cfg_dir: &Path,
    models: &[GenerationConfig],
) -> crate::Result<HashMap<String, Arc<FinalModelConfig>>> {
    let mut model_cfgs = HashMap::new();

    for model in models {
        let model_cfg_path =
            resolve_model_cfg_path(config_dir, infer_cfg_dir, model.model_cfg.as_str());
        let mut raw_model_cfg: RawModelConfig = read_toml_file(&model_cfg_path)?;
        raw_model_cfg.fill_default();

        let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
        model_cfg_builder.fill_auto_after_load();
        model_cfgs.insert(model.model_name.clone(), model_cfg_builder.build_local());
    }

    Ok(model_cfgs)
}

fn resolve_runtime_models(
    infer_cfg_dir: &Path,
    models: &[GenerationConfig],
) -> Vec<GenerationConfig> {
    let mut resolved_models = models.to_vec();
    for model in resolved_models.iter_mut() {
        model.weights_path = resolve_path(infer_cfg_dir, &model.weights_path);
        model.tokenizer_vocab_path = resolve_path(infer_cfg_dir, &model.tokenizer_vocab_path);
    }
    resolved_models
}

fn find_changed_models(
    old_models: &[GenerationConfig],
    new_models: &[GenerationConfig],
) -> Vec<String> {
    let old_map: HashMap<_, _> = old_models
        .iter()
        .map(|model| (model.model_name.clone(), model))
        .collect();
    let new_map: HashMap<_, _> = new_models
        .iter()
        .map(|model| (model.model_name.clone(), model))
        .collect();

    let mut names = HashSet::new();
    for name in old_map.keys() {
        names.insert(name.clone());
    }
    for name in new_map.keys() {
        names.insert(name.clone());
    }

    let mut changed = Vec::new();
    for name in names {
        let old = old_map.get(&name).copied();
        let new = new_map.get(&name).copied();
        if old != new {
            changed.push(name);
        }
    }
    changed.sort();
    changed
}

fn apply_models_patch(
    current: &RawInferConfig,
    patch: &ModelsReloadPatch,
) -> crate::Result<RawInferConfig> {
    let mut next = current.clone();

    let remove_set: HashSet<String> = patch
        .remove_model_names
        .iter()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect();

    let mut upsert_map: HashMap<String, GenerationConfig> = HashMap::new();
    let mut upsert_order: Vec<String> = Vec::new();
    for mut model in patch.upsert.clone() {
        model.fill_default();
        let name = model.model_name.trim().to_string();
        if name.is_empty() {
            return Err(crate::Error::bad_request(
                "model_name cannot be empty in upsert",
            ));
        }
        model.model_name = name.clone();
        if !upsert_map.contains_key(&name) {
            upsert_order.push(name.clone());
        }
        upsert_map.insert(name, model);
    }

    let mut merged = Vec::new();
    for model in &next.models {
        if remove_set.contains(&model.model_name) {
            continue;
        }
        if let Some(replacement) = upsert_map.remove(&model.model_name) {
            merged.push(replacement);
        } else {
            merged.push(model.clone());
        }
    }

    for name in upsert_order {
        if let Some(model) = upsert_map.remove(&name) {
            merged.push(model);
        }
    }

    next.models = merged;
    next.fill_default();
    validate_generation_models(&next.models)?;

    Ok(next)
}

fn write_infer_cfg_atomic(path: &Path, cfg: &RawInferConfig) -> crate::Result<()> {
    let content = toml::to_string_pretty(cfg)
        .map_err(|e| crate::Error::internal_with_source("failed to serialize infer config", e))?;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| {
        crate::Error::internal_with_source(
            format!("failed to create config dir {}", parent.display()),
            e,
        )
    })?;

    let tmp_path = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("infer.toml"),
        std::process::id()
    ));

    fs::write(&tmp_path, content).map_err(|e| {
        crate::Error::internal_with_source(
            format!("failed to write temp config {}", tmp_path.display()),
            e,
        )
    })?;

    if let Err(e) = fs::rename(&tmp_path, path) {
        // Windows may fail to overwrite existing files with rename; retry with explicit remove.
        if path.exists() {
            fs::remove_file(path).map_err(|remove_err| {
                crate::Error::internal(format!(
                    "failed to replace infer config {}: rename_error={e}, remove_error={remove_err}",
                    path.display(),
                ))
            })?;
            fs::rename(&tmp_path, path).map_err(|rename_err| {
                crate::Error::internal_with_source(
                    format!("failed to finalize infer config {}", path.display()),
                    rename_err,
                )
            })?;
        } else {
            return Err(crate::Error::internal(format!(
                "failed to finalize infer config {}: {e}",
                path.display()
            )));
        }
    }

    Ok(())
}

pub trait ModelEngineFactory: Send + Sync {
    fn build_model_groups(
        &self,
        models: &[GenerationConfig],
        model_cfgs: &HashMap<String, Arc<FinalModelConfig>>,
    ) -> crate::Result<HashMap<String, ModelRuntimeGroup>>;
}

#[derive(Clone, Debug, Default)]
pub struct ModelsReloadPatch {
    pub upsert: Vec<GenerationConfig>,
    pub remove_model_names: Vec<String>,
    pub dry_run: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ModelsReloadResult {
    pub changed_model_names: Vec<String>,
    pub rebuilt_model_names: Vec<String>,
    pub removed_model_names: Vec<String>,
    pub active_model_names: Vec<String>,
    pub dry_run: bool,
    pub message: String,
}

pub struct RuntimeManager {
    config_dir: PathBuf,
    infer_cfg_name: String,
    infer_cfg_path: PathBuf,
    raw_infer_cfg: RwLock<RawInferConfig>,
    service: RwLock<Arc<Service>>,
    factory: Arc<dyn ModelEngineFactory>,
    reload_lock: Mutex<()>,
}

impl RuntimeManager {
    pub fn bootstrap(
        config_dir: PathBuf,
        infer_cfg_name: String,
        factory: Arc<dyn ModelEngineFactory>,
    ) -> crate::Result<Self> {
        let infer_cfg_path = config_dir
            .join("infer")
            .join(format!("{infer_cfg_name}.toml"));
        let infer_cfg_dir = infer_cfg_path.parent().unwrap_or_else(|| Path::new("."));

        let raw_infer_cfg = load_raw_infer_cfg(&infer_cfg_path)?;
        let runtime_models = resolve_runtime_models(infer_cfg_dir, &raw_infer_cfg.models);
        let model_cfgs = load_model_cfgs(&config_dir, infer_cfg_dir, &raw_infer_cfg.models)?;
        let groups = factory.build_model_groups(&runtime_models, &model_cfgs)?;
        let service = Arc::new(Service::new(groups)?);

        #[cfg(feature = "trace")]
        tracing::info!(
            infer_cfg = %infer_cfg_name,
            model_count = raw_infer_cfg.models.len(),
            "runtime manager bootstrapped"
        );

        Ok(Self {
            config_dir,
            infer_cfg_name,
            infer_cfg_path,
            raw_infer_cfg: RwLock::new(raw_infer_cfg),
            service: RwLock::new(service),
            factory,
            reload_lock: Mutex::new(()),
        })
    }

    pub fn infer_cfg_name(&self) -> &str {
        &self.infer_cfg_name
    }

    pub fn current_service(&self) -> Arc<Service> {
        self.service
            .read()
            .expect("runtime manager service lock poisoned")
            .clone()
    }

    pub fn model_names(&self) -> Vec<String> {
        self.current_service().model_names()
    }

    pub fn http_bind_addr(&self) -> String {
        self.raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .http_bind_addr
            .clone()
            .unwrap_or_else(|| "0.0.0.0:8080".to_string())
    }

    pub fn request_body_limit_bytes(&self) -> usize {
        self.raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .request_body_limit_bytes
            .unwrap_or(50 * 1024 * 1024)
    }

    pub fn sse_keep_alive_ms(&self) -> u64 {
        self.raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .sse_keep_alive_ms
            .unwrap_or(10_000)
    }

    pub fn allowed_origins(&self) -> Option<Vec<String>> {
        self.raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .allowed_origins
            .clone()
    }

    pub fn api_key(&self) -> Option<String> {
        self.raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .api_key
            .clone()
    }

    fn resolved_ipc_cfg(&self) -> RawIpcConfig {
        let mut cfg = self
            .raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .ipc
            .clone()
            .unwrap_or_default();
        cfg.fill_default();
        cfg
    }

    pub fn ipc_enabled(&self) -> bool {
        self.resolved_ipc_cfg().enabled.unwrap_or(false)
    }

    pub fn ipc_service_name(&self) -> String {
        self.resolved_ipc_cfg()
            .service_name
            .unwrap_or_else(|| "rwkv.infer.openai".to_string())
    }

    pub fn ipc_max_request_bytes(&self) -> usize {
        self.resolved_ipc_cfg()
            .max_request_bytes
            .unwrap_or(4 * 1024 * 1024)
    }

    pub fn ipc_max_response_bytes(&self) -> usize {
        self.resolved_ipc_cfg()
            .max_response_bytes
            .unwrap_or(4 * 1024 * 1024)
    }

    pub fn ipc_max_inflight_requests(&self) -> usize {
        self.resolved_ipc_cfg().max_inflight_requests.unwrap_or(128)
    }

    pub fn ipc_require_api_key(&self) -> bool {
        self.resolved_ipc_cfg().require_api_key.unwrap_or(true)
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "rwkv.infer.runtime.reload_models",
            skip_all,
            fields(
                upsert = patch.upsert.len(),
                remove = patch.remove_model_names.len(),
                dry_run = patch.dry_run
            )
        )
    )]
    pub async fn reload_models(
        &self,
        patch: ModelsReloadPatch,
    ) -> crate::Result<ModelsReloadResult> {
        let _guard = self.reload_lock.lock().await;

        let current_cfg = self
            .raw_infer_cfg
            .read()
            .expect("runtime manager config lock poisoned")
            .clone();
        let next_cfg = apply_models_patch(&current_cfg, &patch)?;

        let changed_model_names = find_changed_models(&current_cfg.models, &next_cfg.models);
        let removed_model_names: Vec<String> = current_cfg
            .models
            .iter()
            .filter(|old| {
                !next_cfg
                    .models
                    .iter()
                    .any(|new| new.model_name == old.model_name)
            })
            .map(|model| model.model_name.clone())
            .collect();

        let changed_set: HashSet<String> = changed_model_names.iter().cloned().collect();
        let rebuild_models: Vec<GenerationConfig> = next_cfg
            .models
            .iter()
            .filter(|model| changed_set.contains(&model.model_name))
            .cloned()
            .collect();

        let mut rebuilt_model_names = Vec::new();
        let current_service = self.current_service();
        let mut merged_groups = current_service.clone_model_groups();

        for removed in &removed_model_names {
            merged_groups.remove(removed);
        }

        if !rebuild_models.is_empty() {
            let infer_cfg_dir = self
                .infer_cfg_path
                .parent()
                .unwrap_or_else(|| Path::new("."));
            let runtime_models = resolve_runtime_models(infer_cfg_dir, &rebuild_models);
            let model_cfgs = load_model_cfgs(&self.config_dir, infer_cfg_dir, &rebuild_models)?;
            let rebuilt_groups = self
                .factory
                .build_model_groups(&runtime_models, &model_cfgs)?;

            rebuilt_model_names = rebuilt_groups.keys().cloned().collect();
            rebuilt_model_names.sort();

            for (model_name, group) in rebuilt_groups {
                merged_groups.insert(model_name, group);
            }
        }

        #[cfg(feature = "trace")]
        tracing::info!(
            changed = changed_model_names.len(),
            rebuilt = rebuilt_model_names.len(),
            removed = removed_model_names.len(),
            "reload patch evaluated"
        );

        let new_service = Arc::new(Service::new(merged_groups)?);

        if !patch.dry_run {
            write_infer_cfg_atomic(&self.infer_cfg_path, &next_cfg)?;
            *self
                .raw_infer_cfg
                .write()
                .expect("runtime manager config lock poisoned") = next_cfg;
            *self
                .service
                .write()
                .expect("runtime manager service lock poisoned") = new_service.clone();
        }

        let active_model_names = if patch.dry_run {
            current_service.model_names()
        } else {
            new_service.model_names()
        };

        let message = if patch.dry_run {
            format!(
                "dry run passed for infer cfg {}. no runtime state changed",
                self.infer_cfg_name
            )
        } else {
            format!(
                "reloaded infer cfg {} and rebuilt {} model group(s)",
                self.infer_cfg_name,
                rebuilt_model_names.len()
            )
        };

        #[cfg(feature = "trace")]
        tracing::info!(
            dry_run = patch.dry_run,
            active = active_model_names.len(),
            "reload models completed"
        );

        Ok(ModelsReloadResult {
            changed_model_names,
            rebuilt_model_names,
            removed_model_names,
            active_model_names,
            dry_run: patch.dry_run,
            message,
        })
    }
}
