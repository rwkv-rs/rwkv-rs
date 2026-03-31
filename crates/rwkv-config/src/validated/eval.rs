use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::raw::eval::{ExtApiConfig, IntApiConfig, SpaceDbConfig};

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::eval::RawEvalConfig", cell = "EVAL_CFG")]
pub struct FinalEvalConfig {
    pub experiment_name: String,
    pub experiment_desc: String,
    pub admin_api_key: Option<String>,
    pub run_mode: String,
    pub skip_checker: bool,
    pub judger_concurrency: usize,
    pub checker_concurrency: usize,
    pub db_pool_max_connections: u32,
    pub model_arch_versions: Vec<String>,
    pub model_data_versions: Vec<String>,
    pub model_num_params: Vec<String>,
    pub benchmark_field: Vec<String>,
    pub extra_benchmark_name: Vec<String>,
    pub upload_to_space: bool,
    pub git_hash: String,
    pub models: Vec<IntApiConfig>,
    pub llm_judger: ExtApiConfig,
    pub llm_checker: ExtApiConfig,
    pub space_db: Option<SpaceDbConfig>,
}

pub static EVAL_CFG: OnceCell<Arc<FinalEvalConfig>> = OnceCell::new();
