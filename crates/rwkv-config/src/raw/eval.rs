use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RawEvalConfig {
    pub experiment_name: String,
    pub experiment_desc: String,
    pub run_mode: Option<String>,
    pub continue_on_benchmark_error: Option<bool>,
    pub skip_checker: Option<bool>,
    pub attempt_concurrency: Option<usize>,
    pub judger_concurrency: Option<usize>,
    pub checker_concurrency: Option<usize>,
    pub db_pool_max_connections: Option<u32>,
    pub sample_limit: Option<usize>,
    pub browsecomp_cot_max_tokens: Option<u32>,
    pub browsecomp_answer_max_tokens: Option<u32>,
    pub model_arch_versions: Vec<String>,
    pub model_data_versions: Vec<String>,
    pub model_num_params: Vec<String>,
    pub benchmark_field: Vec<String>,
    pub extra_benchmark_name: Vec<String>,
    pub upload_to_space: Option<bool>,
    pub startup_recovery: Option<bool>,
    pub git_hash: String,
    pub models: Vec<IntApiConfig>,
    pub llm_judger: ExtApiConfig,
    pub llm_checker: ExtApiConfig,
    pub space_db: SpaceDbConfig,
}

impl RawEvalConfig {
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            upload_to_space: false,
            startup_recovery: false,
            run_mode: "new".to_string(),
            continue_on_benchmark_error: false,
            skip_checker: false,
            attempt_concurrency: 8,
            judger_concurrency: 8,
            checker_concurrency: 8,
            db_pool_max_connections: 32,
            browsecomp_cot_max_tokens: 2048,
            browsecomp_answer_max_tokens: 1024
        );
        self.space_db.fill_default();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct IntApiConfig {
    pub model_arch_version: String,
    pub model_data_version: String,
    pub model_num_params: String,
    pub base_url: String,
    #[serde(skip_serializing)]
    pub api_key: String,
    pub model: String,
    pub max_batch_size: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExtApiConfig {
    pub base_url: String,
    #[serde(skip_serializing)]
    pub api_key: String,
    pub model: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpaceDbConfig {
    pub username: String,
    #[serde(skip_serializing)]
    pub password: String,
    pub host: String,
    pub port: String,
    pub database_name: String,
    pub sslmode: Option<String>,
}

impl SpaceDbConfig {
    pub fn fill_default(&mut self) {
        fill_default!(self, sslmode: "verify-full".to_string());
    }
}
