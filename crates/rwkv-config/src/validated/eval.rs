use std::collections::HashSet;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::raw::eval::{EvalModelConfig, LlmServiceConfig, SpaceDbConfig};

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
    pub models: Vec<EvalModelConfig>,
    pub llm_judger: LlmServiceConfig,
    pub llm_checker: LlmServiceConfig,
    pub space_db: Option<SpaceDbConfig>,
}

impl FinalEvalConfigBuilder {
    pub fn check(&self) {
        assert!(
            !self.get_experiment_name().unwrap().trim().is_empty(),
            "experiment_name cannot be empty"
        );
        assert!(
            !self.get_experiment_desc().unwrap().trim().is_empty(),
            "experiment_desc cannot be empty"
        );
        assert!(
            !self.get_git_hash().unwrap().trim().is_empty(),
            "git_hash cannot be empty"
        );
        assert!(
            !self.get_model_arch_versions().unwrap().is_empty(),
            "model_arch_versions cannot be empty"
        );
        assert!(
            !self.get_model_data_versions().unwrap().is_empty(),
            "model_data_versions cannot be empty"
        );
        assert!(
            !self.get_model_num_params().unwrap().is_empty(),
            "model_num_params cannot be empty"
        );

        let models = self.get_models().unwrap();
        assert!(!models.is_empty(), "models cannot be empty");

        let mut model_names = HashSet::new();
        for model in models {
            assert!(
                !model.model_arch_version.trim().is_empty(),
                "model_arch_version cannot be empty"
            );
            assert!(
                !model.model_data_version.trim().is_empty(),
                "model_data_version cannot be empty"
            );
            assert!(
                !model.model_num_params.trim().is_empty(),
                "model_num_params cannot be empty"
            );
            assert!(
                !model.base_url.trim().is_empty(),
                "base_url cannot be empty"
            );
            assert!(!model.api_key.trim().is_empty(), "api_key cannot be empty");
            assert!(!model.model.trim().is_empty(), "model cannot be empty");
            assert!(
                model.max_batch_size.unwrap() >= 1,
                "max_batch_size must be >= 1 for model {}",
                model.model
            );
            assert!(
                model_names.insert(model.model.clone()),
                "duplicated model: {}",
                model.model
            );
        }

        check_service(&self.get_llm_judger().unwrap(), "llm_judger");
        check_service(&self.get_llm_checker().unwrap(), "llm_checker");
    }
}

fn check_service(service: &LlmServiceConfig, field_name: &str) {
    assert!(
        !service.base_url.trim().is_empty(),
        "{field_name}.base_url cannot be empty"
    );
    assert!(
        !service.api_key.trim().is_empty(),
        "{field_name}.api_key cannot be empty"
    );
    assert!(
        !service.model.trim().is_empty(),
        "{field_name}.model cannot be empty"
    );
}

pub static EVAL_CFG: OnceCell<Arc<FinalEvalConfig>> = OnceCell::new();
