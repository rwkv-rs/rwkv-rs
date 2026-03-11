use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::raw::eval::{ApiConfig, SpaceDbConfig};

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::eval::RawEvalConfig", cell = "EVAL_CFG")]
pub struct FinalEvalConfig {
    pub experiment_name: String,
    pub experiment_desc: String,
    pub model_arch_versions: Vec<String>,
    pub model_data_versions: Vec<String>,
    pub model_num_params: Vec<String>,
    pub benchmark_field: Vec<String>,
    pub extra_benchmark_name: Vec<String>,
    pub upload_to_space: bool,
    pub git_hash: String,
    pub models: Vec<ApiConfig>,
    pub llm_judger: ApiConfig,
    pub llm_checker: ApiConfig,
    pub space_db: Option<SpaceDbConfig>,
}

pub static EVAL_CFG: OnceCell<Arc<FinalEvalConfig>> = OnceCell::new();
