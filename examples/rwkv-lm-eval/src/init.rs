use std::path::{Path, PathBuf};

use rwkv_config::{load_toml, raw::eval::RawEvalConfig, validated::eval::FinalEvalConfigBuilder};
use rwkv_eval::{routes::http_api::AppState, services::runner::build_http_app_state};

pub fn resolve_eval_cfg_path(config_dir: &Path, eval_cfg_name: &str) -> PathBuf {
    [
        config_dir
            .join("eval")
            .join(format!("{eval_cfg_name}.toml")),
        config_dir.join(format!("{eval_cfg_name}.toml")),
        PathBuf::from("examples")
            .join("rwkv-lm-eval")
            .join("config")
            .join(format!("{eval_cfg_name}.toml")),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .unwrap_or_else(|| {
        panic!(
            "failed to locate eval config `{}` under {}",
            eval_cfg_name,
            config_dir.display()
        )
    })
}

pub fn load_raw_eval_cfg(path: &Path) -> RawEvalConfig {
    let mut raw_eval_cfg: RawEvalConfig = load_toml(path);
    raw_eval_cfg.fill_default();
    raw_eval_cfg
}

pub fn build_eval_cfg_builder(path: &Path) -> FinalEvalConfigBuilder {
    FinalEvalConfigBuilder::load_from_raw(load_raw_eval_cfg(path))
}

pub async fn build_http_runtime(
    config_dir: &Path,
    eval_cfg_name: &str,
    db_pool_max_connections: Option<u32>,
) -> (AppState, PathBuf, u32) {
    let config_path = resolve_eval_cfg_path(config_dir, eval_cfg_name);
    let raw_eval_cfg = load_raw_eval_cfg(&config_path);
    let max_connections = db_pool_max_connections
        .or(raw_eval_cfg.db_pool_max_connections)
        .unwrap_or(32);
    let app_state = build_http_app_state(raw_eval_cfg, db_pool_max_connections)
        .await
        .unwrap_or_else(|err| panic!("failed to prepare eval app state: {err}"));

    (app_state, config_path, max_connections)
}
