use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clia_tracing_config::WorkerGuard;
use rwkv_config::load_toml;
use rwkv_config::raw::infer::RawInferConfig;
use rwkv_config::raw::model::RawModelConfig;
use rwkv_config::validated::infer::FinalInferConfigBuilder;
use rwkv_config::validated::model::{FinalModelConfig, FinalModelConfigBuilder};

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

pub fn init_cfg<P: AsRef<Path>>(
    config_dir: P,
    infer_cfg_name: &str,
) -> (
    FinalInferConfigBuilder,
    HashMap<String, Arc<FinalModelConfig>>,
) {
    let config_dir = config_dir.as_ref();
    let infer_cfg_path = config_dir
        .join("infer")
        .join(format!("{infer_cfg_name}.toml"));
    let infer_cfg_dir = infer_cfg_path.parent().unwrap_or_else(|| Path::new("."));

    let mut raw_infer_cfg: RawInferConfig = load_toml(&infer_cfg_path);
    raw_infer_cfg.fill_default();

    // Resolve relative paths against the infer config directory.
    for model in raw_infer_cfg.models.iter_mut() {
        model.weights_path = resolve_path(infer_cfg_dir, &model.weights_path);
        model.tokenizer_vocab_path = resolve_path(infer_cfg_dir, &model.tokenizer_vocab_path);
    }

    let mut model_cfgs: HashMap<String, Arc<FinalModelConfig>> = HashMap::new();
    for model in raw_infer_cfg.models.iter() {
        let model_cfg_path =
            resolve_model_cfg_path(config_dir, infer_cfg_dir, model.model_cfg.as_str());
        let mut raw_model_cfg: RawModelConfig = load_toml(&model_cfg_path);
        raw_model_cfg.fill_default();

        let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
        model_cfg_builder.fill_auto_after_load();
        model_cfgs.insert(model.model_name.clone(), model_cfg_builder.build_local());
    }

    let infer_cfg_builder = FinalInferConfigBuilder::load_from_raw(raw_infer_cfg);
    infer_cfg_builder.check();

    (infer_cfg_builder, model_cfgs)
}

pub fn init_cfg_paths<P1: AsRef<Path>, P2: AsRef<Path>>(
    model_cfg_path: P1,
    infer_cfg_path: P2,
) -> (FinalModelConfigBuilder, FinalInferConfigBuilder) {
    let mut raw_model_cfg: RawModelConfig = load_toml(model_cfg_path);
    let mut raw_infer_cfg: RawInferConfig = load_toml(infer_cfg_path);
    raw_model_cfg.fill_default();
    raw_infer_cfg.fill_default();

    let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
    let infer_cfg_builder = FinalInferConfigBuilder::load_from_raw(raw_infer_cfg);

    model_cfg_builder.fill_auto_after_load();
    (model_cfg_builder, infer_cfg_builder)
}

pub fn init_log(level: &str) -> WorkerGuard {
    clia_tracing_config::build()
        .filter_level(level)
        .with_ansi(true)
        .to_stdout(true)
        .init()
}
