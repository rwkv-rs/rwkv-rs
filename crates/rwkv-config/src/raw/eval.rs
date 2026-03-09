use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RawEvalConfig {
    pub experiment_name: String,
    pub experiment_desc: String,
    pub model_arch_versions: Vec<String>,
    pub model_data_versions: Vec<String>,
    pub model_num_params: Vec<String>,
    pub benchmark_field: Vec<String>,
    pub extra_benchmark_name: Vec<String>,
    pub upload_to_space: Option<bool>,
    pub git_hash: String,
    pub models: Vec<ApiConfig>,
    pub llm_judger: ApiConfig,
    pub llm_checker: ApiConfig,
    pub space_db: SpaceDbConfig,
}

impl RawEvalConfig {
    pub fn fill_default(&mut self) {
        fill_default!(self, upload_to_space: false);
        self.space_db.fill_default();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ApiConfig {
    pub model_arch_version: String,
    pub model_data_version: String,
    pub model_num_params: String,
    pub base_url: String,
    #[serde(skip_serializing)]
    pub api_key: String,
    pub model: String,
    pub max_batch_size: Option<usize>,
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
