use std::env;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use once_cell::sync::Lazy;
use rwkv_config::{
    load_toml,
    raw::eval::{ExtApiConfig, RawEvalConfig},
    validated::eval::FinalEvalConfigBuilder,
};

#[derive(Clone, Debug, Default)]
pub struct RuntimeExtApiConfigOverrides {
    pub llm_judger: Option<ExtApiConfig>,
    pub llm_checker: Option<ExtApiConfig>,
}

static RUNTIME_EXT_API_CONFIG_OVERRIDES: Lazy<RwLock<RuntimeExtApiConfigOverrides>> =
    Lazy::new(|| RwLock::new(RuntimeExtApiConfigOverrides::default()));

fn candidate_eval_cfg_paths(config_dir: &Path, eval_cfg_name: &str) -> [PathBuf; 3] {
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
}

fn read_env_nonempty(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn is_env_override_required(value: &str) -> bool {
    value.trim() == "env_override_required"
}

fn apply_eval_env_overrides(raw_eval_cfg: &mut RawEvalConfig) {
    let shared_ext_model = read_env_nonempty(&["RWKV_EVAL_EXT_MODEL"]);
    let shared_ext_base_url = read_env_nonempty(&["RWKV_EVAL_EXT_BASE_URL"]);
    let shared_ext_api_key = read_env_nonempty(&["RWKV_EVAL_EXT_API_KEY"]);
    let target_model_override =
        read_env_nonempty(&["RWKV_EVAL_TARGET_MODEL"]).or_else(|| shared_ext_model.clone());
    let target_base_url_override =
        read_env_nonempty(&["RWKV_EVAL_TARGET_BASE_URL"]).or_else(|| shared_ext_base_url.clone());
    let target_api_key_override =
        read_env_nonempty(&["RWKV_EVAL_TARGET_API_KEY"]).or_else(|| shared_ext_api_key.clone());

    for model_cfg in &mut raw_eval_cfg.models {
        if is_env_override_required(&model_cfg.model) {
            if let Some(model) = target_model_override.clone() {
                model_cfg.model = model;
            }
        }
        if is_env_override_required(&model_cfg.base_url) {
            if let Some(base_url) = target_base_url_override.clone() {
                model_cfg.base_url = base_url;
            }
        }
        if is_env_override_required(&model_cfg.api_key) {
            if let Some(api_key) = target_api_key_override.clone() {
                model_cfg.api_key = api_key;
            }
        }
    }

    if let Some(model) =
        read_env_nonempty(&["RWKV_EVAL_LLM_JUDGER_MODEL"]).or_else(|| shared_ext_model.clone())
    {
        raw_eval_cfg.llm_judger.model = model;
    }
    if let Some(base_url) = read_env_nonempty(&["RWKV_EVAL_LLM_JUDGER_BASE_URL"])
        .or_else(|| shared_ext_base_url.clone())
    {
        raw_eval_cfg.llm_judger.base_url = base_url;
    }
    if let Some(api_key) =
        read_env_nonempty(&["RWKV_EVAL_LLM_JUDGER_API_KEY"]).or_else(|| shared_ext_api_key.clone())
    {
        raw_eval_cfg.llm_judger.api_key = api_key;
    }

    if let Some(model) =
        read_env_nonempty(&["RWKV_EVAL_LLM_CHECKER_MODEL"]).or_else(|| shared_ext_model)
    {
        raw_eval_cfg.llm_checker.model = model;
    }
    if let Some(base_url) =
        read_env_nonempty(&["RWKV_EVAL_LLM_CHECKER_BASE_URL"]).or_else(|| shared_ext_base_url)
    {
        raw_eval_cfg.llm_checker.base_url = base_url;
    }
    if let Some(api_key) =
        read_env_nonempty(&["RWKV_EVAL_LLM_CHECKER_API_KEY"]).or_else(|| shared_ext_api_key)
    {
        raw_eval_cfg.llm_checker.api_key = api_key;
    }

    if let Some(host) = read_env_nonempty(&["RWKV_EVAL_SPACE_DB_HOST"]) {
        raw_eval_cfg.space_db.host = host;
    }
    if let Some(port) = read_env_nonempty(&["RWKV_EVAL_SPACE_DB_PORT"]) {
        raw_eval_cfg.space_db.port = port;
    }
    if let Some(username) = read_env_nonempty(&["RWKV_EVAL_SPACE_DB_USERNAME"]) {
        raw_eval_cfg.space_db.username = username;
    }
    if let Some(password) = read_env_nonempty(&["RWKV_EVAL_SPACE_DB_PASSWORD"]) {
        raw_eval_cfg.space_db.password = password;
    }
    if let Some(database_name) = read_env_nonempty(&["RWKV_EVAL_SPACE_DB_DATABASE_NAME"]) {
        raw_eval_cfg.space_db.database_name = database_name;
    }
    if let Some(sslmode) = read_env_nonempty(&["RWKV_EVAL_SPACE_DB_SSLMODE"]) {
        raw_eval_cfg.space_db.sslmode = Some(sslmode);
    }
}

pub fn set_runtime_ext_api_config_overrides(
    llm_judger: Option<ExtApiConfig>,
    llm_checker: Option<ExtApiConfig>,
) {
    let mut guard = RUNTIME_EXT_API_CONFIG_OVERRIDES
        .write()
        .unwrap_or_else(|err| panic!("runtime ext api config overrides lock poisoned: {err}"));
    guard.llm_judger = llm_judger;
    guard.llm_checker = llm_checker;
}

pub fn runtime_ext_api_config_overrides() -> RuntimeExtApiConfigOverrides {
    RUNTIME_EXT_API_CONFIG_OVERRIDES
        .read()
        .unwrap_or_else(|err| panic!("runtime ext api config overrides lock poisoned: {err}"))
        .clone()
}

pub fn init_cfg<P: AsRef<Path>>(config_dir: P, eval_cfg_name: &str) -> FinalEvalConfigBuilder {
    let config_dir = config_dir.as_ref();
    let eval_cfg_path = candidate_eval_cfg_paths(config_dir, eval_cfg_name)
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| {
            panic!(
                "failed to locate eval config `{}` under {}",
                eval_cfg_name,
                config_dir.display()
            )
        });

    let mut raw_eval_cfg: RawEvalConfig = load_toml(&eval_cfg_path);
    raw_eval_cfg.fill_default();
    apply_eval_env_overrides(&mut raw_eval_cfg);

    FinalEvalConfigBuilder::load_from_raw(raw_eval_cfg)
}
