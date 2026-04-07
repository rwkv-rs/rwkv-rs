use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RawEvalConfig {
    pub experiment_name: String,
    pub experiment_desc: String,
    pub admin_api_key: Option<String>,
    pub run_mode: Option<String>,
    pub skip_checker: Option<bool>,
    pub skip_dataset_check: Option<bool>,
    pub judger_concurrency: Option<usize>,
    pub checker_concurrency: Option<usize>,
    pub db_pool_max_connections: Option<u32>,
    pub model_arch_versions: Vec<String>,
    pub model_data_versions: Vec<String>,
    pub model_num_params: Vec<String>,
    pub benchmark_field: Vec<String>,
    pub extra_benchmark_name: Vec<String>,
    pub upload_to_space: Option<bool>,
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
            run_mode: "new".to_string(),
            skip_checker: false,
            skip_dataset_check: false,
            judger_concurrency: 8,
            checker_concurrency: 8,
            db_pool_max_connections: 32
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
    pub api_key: String,
    pub model: String,
    pub max_batch_size: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExtApiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct SpaceDbConfig {
    pub username: String,
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

#[cfg(test)]
mod tests {
    use super::{ExtApiConfig, IntApiConfig, RawEvalConfig, SpaceDbConfig};

    fn sample_cfg() -> RawEvalConfig {
        RawEvalConfig {
            experiment_name: "demo".to_string(),
            experiment_desc: "demo".to_string(),
            admin_api_key: Some("admin-secret".to_string()),
            run_mode: Some("new".to_string()),
            skip_checker: Some(false),
            skip_dataset_check: Some(false),
            judger_concurrency: Some(8),
            checker_concurrency: Some(8),
            db_pool_max_connections: Some(32),
            model_arch_versions: vec!["rwkv7".to_string()],
            model_data_versions: vec!["g1".to_string()],
            model_num_params: vec!["1.5b".to_string()],
            benchmark_field: vec!["Knowledge".to_string()],
            extra_benchmark_name: Vec::new(),
            upload_to_space: Some(true),
            git_hash: "abc123".to_string(),
            models: vec![IntApiConfig {
                model_arch_version: "rwkv7".to_string(),
                model_data_version: "g1".to_string(),
                model_num_params: "1.5b".to_string(),
                base_url: "http://127.0.0.1:8000".to_string(),
                api_key: "model-secret".to_string(),
                model: "rwkv7-g1-1.5b".to_string(),
                max_batch_size: Some(32),
            }],
            llm_judger: ExtApiConfig {
                base_url: "http://127.0.0.1:9000".to_string(),
                api_key: "judger-secret".to_string(),
                model: "judge".to_string(),
            },
            llm_checker: ExtApiConfig {
                base_url: "http://127.0.0.1:9001".to_string(),
                api_key: "checker-secret".to_string(),
                model: "checker".to_string(),
            },
            space_db: SpaceDbConfig {
                username: "postgres".to_string(),
                password: "db-secret".to_string(),
                host: "localhost".to_string(),
                port: "5432".to_string(),
                database_name: "rwkv".to_string(),
                sslmode: Some("verify-full".to_string()),
            },
        }
    }

    #[test]
    fn eval_config_roundtrip_preserves_sensitive_fields() {
        let raw = toml::to_string_pretty(&sample_cfg()).unwrap();

        assert!(raw.contains("admin_api_key = \"admin-secret\""));
        assert!(raw.contains("api_key = \"model-secret\""));
        assert!(raw.contains("api_key = \"judger-secret\""));
        assert!(raw.contains("api_key = \"checker-secret\""));
        assert!(raw.contains("password = \"db-secret\""));

        let decoded: RawEvalConfig = toml::from_str(&raw).unwrap();
        assert_eq!(decoded, sample_cfg());
    }
}
