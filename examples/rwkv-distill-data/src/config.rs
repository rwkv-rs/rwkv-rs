use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Result, ensure};
use serde::Deserialize;

macro_rules! default_value {
    ($name:ident, $ty:ty, $value:expr) => {
        fn $name() -> $ty {
            $value
        }
    };
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub input: InputConfig,
    pub generator: GeneratorConfig,
    pub answer_models: Vec<ModelConfig>,
    #[serde(default = "OutputConfig::default")]
    pub output: OutputConfig,
    #[serde(default = "RunConfig::default")]
    pub run: RunConfig,
    #[serde(default = "ConcurrencyConfig::default")]
    pub concurrency: ConcurrencyConfig,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputConfig {
    pub dataset_path: PathBuf,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub start_index: usize,
    #[serde(default = "default_subject", rename = "default_subject")]
    pub _default_subject: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorConfig {
    #[serde(flatten)]
    pub model: ModelConfig,
    pub variant_count: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub endpoint: String,
    pub model_name: String,
    pub api_key: String,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    #[serde(default = "default_state_jsonl_path")]
    pub state_jsonl_path: PathBuf,
    #[serde(
        default = "default_train_jsonl_path",
        alias = "jsonl_path",
        alias = "train_jsonl_path"
    )]
    pub train_jsonl_path: PathBuf,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    #[serde(default = "default_resume")]
    pub resume: bool,
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: f64,
    #[serde(default = "default_disable_env_proxy")]
    pub disable_env_proxy: bool,
    #[serde(default = "default_force_http1")]
    pub force_http1: bool,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcurrencyConfig {
    #[serde(default = "default_generate_requests")]
    pub generate_requests: usize,
    #[serde(default = "default_answer_requests")]
    pub answer_requests: usize,
}

pub fn load_config(path: &Path) -> Result<Config> {
    let raw = fs::read_to_string(path)?;
    let mut cfg: Config = toml::from_str(&raw)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    cfg.input.dataset_path = resolve(base, &cfg.input.dataset_path);
    cfg.output.state_jsonl_path = resolve(base, &cfg.output.state_jsonl_path);
    cfg.output.train_jsonl_path = resolve(base, &cfg.output.train_jsonl_path);
    Ok(cfg)
}

pub fn validate_config(cfg: &Config) -> Result<()> {
    ensure!(
        cfg.generator.variant_count > 0,
        "generator.variant_count must be > 0"
    );
    ensure!(
        !cfg.answer_models.is_empty(),
        "need at least one answer model"
    );
    ensure!(
        cfg.run.request_timeout_seconds > 0.0,
        "run.request_timeout_seconds must be > 0"
    );
    validate_model(&cfg.generator.model)?;
    cfg.answer_models.iter().try_for_each(validate_model)?;
    Ok(())
}

fn validate_model(model: &ModelConfig) -> Result<()> {
    ensure!(
        !model.endpoint.trim().is_empty(),
        "model endpoint must not be empty"
    );
    ensure!(
        !model.model_name.trim().is_empty(),
        "model_name must not be empty"
    );
    ensure!(
        !model.api_key.trim().is_empty(),
        "api_key must not be empty"
    );
    Ok(())
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

default_value!(default_subject, String, "general".to_owned());
default_value!(
    default_state_jsonl_path,
    PathBuf,
    PathBuf::from("data/distill_state.jsonl")
);
default_value!(
    default_train_jsonl_path,
    PathBuf,
    PathBuf::from("data/rwkv_train.jsonl")
);
default_value!(default_resume, bool, true);
default_value!(default_request_timeout_seconds, f64, 240.0);
default_value!(default_disable_env_proxy, bool, true);
default_value!(default_force_http1, bool, true);
default_value!(default_generate_requests, usize, 4);
default_value!(default_answer_requests, usize, 16);

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            state_jsonl_path: default_state_jsonl_path(),
            train_jsonl_path: default_train_jsonl_path(),
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            resume: default_resume(),
            request_timeout_seconds: default_request_timeout_seconds(),
            disable_env_proxy: default_disable_env_proxy(),
            force_http1: default_force_http1(),
        }
    }
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            generate_requests: default_generate_requests(),
            answer_requests: default_answer_requests(),
        }
    }
}
