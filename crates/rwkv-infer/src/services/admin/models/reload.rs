use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    sync::Arc,
};

use serde::de::DeserializeOwned;
use rwkv_config::raw::infer::{GenerationConfig, RawInferConfig};

use crate::{
    cores::queue::queue_worker::QueueHandle,
    dtos::admin::models::reload::{ModelsReloadReq, ModelsReloadResp},
    services::{QueueMap, QueueMapBuilder, ServiceError, ServiceResult, SharedQueueMap},
};

pub async fn reload_models(
    queues: &SharedQueueMap,
    reload_lock: Option<&Arc<tokio::sync::Mutex<()>>>,
    infer_cfg_path: Option<&Path>,
    build_queues: Option<&QueueMapBuilder>,
    req: ModelsReloadReq,
) -> ServiceResult<ModelsReloadResp> {
    let Some(reload_lock) = reload_lock else {
        return Err(ServiceError::not_supported(
            "model reload is not configured for this server",
        ));
    };
    let Some(infer_cfg_path) = infer_cfg_path else {
        return Err(ServiceError::not_supported(
            "reload infer config path is not configured",
        ));
    };
    let Some(build_queues) = build_queues else {
        return Err(ServiceError::not_supported(
            "reload queue builder is not configured",
        ));
    };

    let _guard = reload_lock.lock().await;

    let current_cfg = load_raw_infer_cfg(infer_cfg_path)?;
    let next_cfg = apply_models_patch(&current_cfg, &req)?;

    let changed_model_names = find_changed_models(&current_cfg.models, &next_cfg.models);
    let changed_set: HashSet<String> = changed_model_names.iter().cloned().collect();
    let removed_model_names: Vec<String> = current_cfg
        .models
        .iter()
        .filter(|old_model| {
            !next_cfg
                .models
                .iter()
                .any(|new_model| new_model.model_name == old_model.model_name)
        })
        .map(|model| model.model_name.clone())
        .collect();

    let rebuild_models: Vec<GenerationConfig> = next_cfg
        .models
        .iter()
        .filter(|model| changed_set.contains(&model.model_name))
        .cloned()
        .collect();

    let rebuilt_queues = if rebuild_models.is_empty() {
        HashMap::new()
    } else {
        (build_queues)(rebuild_models.as_slice())?
    };
    if let Err(err) = validate_rebuilt_queues(&rebuild_models, &rebuilt_queues) {
        shutdown_queue_map(rebuilt_queues).await;
        return Err(err);
    }

    let rebuilt_model_names = {
        let mut names: Vec<String> = rebuilt_queues.keys().cloned().collect();
        names.sort();
        names
    };

    if let Err(err) = write_infer_cfg_atomic(infer_cfg_path, &next_cfg) {
        shutdown_queue_map(rebuilt_queues).await;
        return Err(err);
    }

    let current_queue_map = queues
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let mut merged_queues = current_queue_map.clone();
    let mut drained_queues = Vec::new();

    for removed_model_name in &removed_model_names {
        if let Some(handles) = merged_queues.remove(removed_model_name) {
            drained_queues.extend(handles);
        }
    }
    for (model_name, handles) in rebuilt_queues {
        if let Some(old_handles) = merged_queues.insert(model_name, handles) {
            drained_queues.extend(old_handles);
        }
    }

    let mut active_model_names: Vec<String> = merged_queues.keys().cloned().collect();
    active_model_names.sort();

    *queues
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = merged_queues;

    for handle in drained_queues {
        handle.begin_drain();
        tokio::spawn(async move {
            handle.shutdown().await;
        });
    }

    Ok(ModelsReloadResp {
        changed_model_names,
        rebuilt_model_names: rebuilt_model_names.clone(),
        removed_model_names: removed_model_names.clone(),
        active_model_names,
        message: format!(
            "reloaded infer config and rebuilt {} model group(s)",
            rebuilt_model_names.len()
        ),
    })
}

async fn shutdown_queue_map(queue_map: QueueMap) {
    for handle in queue_map.into_values().flatten() {
        shutdown_queue_handle(handle).await;
    }
}

async fn shutdown_queue_handle(handle: QueueHandle) {
    handle.begin_drain();
    handle.shutdown().await;
}

fn validate_rebuilt_queues(
    rebuild_models: &[GenerationConfig],
    rebuilt_queues: &QueueMap,
) -> ServiceResult<()> {
    let expected_names: HashSet<String> = rebuild_models
        .iter()
        .map(|model| model.model_name.clone())
        .collect();
    let actual_names: HashSet<String> = rebuilt_queues.keys().cloned().collect();

    if expected_names != actual_names {
        return Err(ServiceError::internal(format!(
            "reload builder returned mismatched models: expected={expected_names:?}, actual={actual_names:?}"
        )));
    }

    for (model_name, handles) in rebuilt_queues {
        if handles.is_empty() {
            return Err(ServiceError::internal(format!(
                "reload builder returned no queues for model {model_name}"
            )));
        }
    }

    Ok(())
}

fn apply_models_patch(
    current: &RawInferConfig,
    req: &ModelsReloadReq,
) -> ServiceResult<RawInferConfig> {
    let mut next = current.clone();

    let remove_set: HashSet<String> = req
        .remove_model_names
        .iter()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect();

    let mut upsert_map: HashMap<String, GenerationConfig> = HashMap::new();
    let mut upsert_order = Vec::new();
    for mut model in req.upsert.clone() {
        model.fill_default();
        let model_name = model.model_name.trim().to_string();
        if model_name.is_empty() {
            return Err(ServiceError::bad_request(
                "model_name cannot be empty in reload upsert",
            ));
        }
        model.model_name = model_name.clone();
        if !upsert_map.contains_key(&model_name) {
            upsert_order.push(model_name.clone());
        }
        upsert_map.insert(model_name, model);
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

    for model_name in upsert_order {
        if let Some(model) = upsert_map.remove(&model_name) {
            merged.push(model);
        }
    }

    next.models = merged;
    next.fill_default();
    validate_generation_models(&next.models)?;
    Ok(next)
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
    names.extend(old_map.keys().cloned());
    names.extend(new_map.keys().cloned());

    let mut changed = Vec::new();
    for name in names {
        if old_map.get(&name) != new_map.get(&name) {
            changed.push(name);
        }
    }
    changed.sort();
    changed
}

fn validate_generation_models(models: &[GenerationConfig]) -> ServiceResult<()> {
    if models.is_empty() {
        return Err(ServiceError::bad_request(
            "infer config requires at least one model",
        ));
    }

    let mut names = HashSet::new();
    for model in models {
        if model.model_name.trim().is_empty() {
            return Err(ServiceError::bad_request("model_name cannot be empty"));
        }
        if !names.insert(model.model_name.clone()) {
            return Err(ServiceError::bad_request(format!(
                "duplicated model_name: {}",
                model.model_name
            )));
        }
        if model.model_cfg.trim().is_empty() {
            return Err(ServiceError::bad_request(format!(
                "model_cfg cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.weights_path.trim().is_empty() {
            return Err(ServiceError::bad_request(format!(
                "weights_path cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.tokenizer_vocab_path.trim().is_empty() {
            return Err(ServiceError::bad_request(format!(
                "tokenizer_vocab_path cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.device_ids.is_empty() {
            return Err(ServiceError::bad_request(format!(
                "device_ids cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.max_batch_size.unwrap_or_default() < 1 {
            return Err(ServiceError::bad_request(format!(
                "max_batch_size must be >= 1 for model {}",
                model.model_name
            )));
        }
        if model.max_context_len.unwrap_or_default() < 1 {
            return Err(ServiceError::bad_request(format!(
                "max_context_len must be >= 1 for model {}",
                model.model_name
            )));
        }
    }

    Ok(())
}

fn load_raw_infer_cfg(path: &Path) -> ServiceResult<RawInferConfig> {
    let mut cfg: RawInferConfig = read_toml_file(path)?;
    cfg.fill_default();
    validate_generation_models(&cfg.models)?;
    Ok(cfg)
}

fn read_toml_file<T: DeserializeOwned>(path: &Path) -> ServiceResult<T> {
    let content = fs::read_to_string(path).map_err(|err| {
        ServiceError::bad_request(format!("failed to read config {}: {err}", path.display()))
    })?;

    toml::from_str(&content)
        .map_err(|err| ServiceError::bad_request(format!("invalid toml {}: {err}", path.display())))
}

fn write_infer_cfg_atomic(path: &Path, cfg: &RawInferConfig) -> ServiceResult<()> {
    let content = toml::to_string_pretty(cfg).map_err(|err| {
        ServiceError::internal(format!("failed to serialize infer config: {err}"))
    })?;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|err| {
        ServiceError::internal(format!(
            "failed to create config dir {}: {err}",
            parent.display()
        ))
    })?;

    let tmp_path = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|file_name| file_name.to_str())
            .unwrap_or("infer.toml"),
        std::process::id()
    ));

    fs::write(&tmp_path, content).map_err(|err| {
        ServiceError::internal(format!(
            "failed to write temp config {}: {err}",
            tmp_path.display()
        ))
    })?;

    if let Err(err) = fs::rename(&tmp_path, path) {
        if path.exists() {
            fs::remove_file(path).map_err(|remove_err| {
                ServiceError::internal(format!(
                    "failed to replace infer config {}: rename_error={err}, remove_error={remove_err}",
                    path.display(),
                ))
            })?;
            fs::rename(&tmp_path, path).map_err(|rename_err| {
                ServiceError::internal(format!(
                    "failed to finalize infer config {}: {rename_err}",
                    path.display()
                ))
            })?;
        } else {
            return Err(ServiceError::internal(format!(
                "failed to finalize infer config {}: {err}",
                path.display()
            )));
        }
    }

    Ok(())
}
