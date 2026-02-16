use std::collections::HashSet;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::raw::infer::RawInferModelConfig;

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::infer::RawInferConfig", cell = "INFER_CFG")]
pub struct FinalInferConfig {
    // HTTP
    pub http_bind_addr: String,
    pub request_body_limit_bytes: usize,
    pub sse_keep_alive_ms: u64,
    pub allowed_origins: Option<Vec<String>>,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,

    // Multi-model deployment
    pub models: Vec<RawInferModelConfig>,
}

impl FinalInferConfigBuilder {
    pub fn check(&self) {
        let models = self.get_models().unwrap();
        assert!(!models.is_empty(), "infer config requires at least one model");

        let mut names = HashSet::new();
        for model in models {
            assert!(
                !model.model_name.trim().is_empty(),
                "model_name cannot be empty"
            );
            assert!(
                names.insert(model.model_name.clone()),
                "duplicated model_name: {}",
                model.model_name
            );
            assert!(
                !model.weights_path.trim().is_empty(),
                "weights_path cannot be empty"
            );
            assert!(
                !model.tokenizer_vocab_path.trim().is_empty(),
                "tokenizer_vocab_path cannot be empty"
            );
            assert!(
                !model.device_ids.is_empty(),
                "device_ids cannot be empty for model {}",
                model.model_name
            );
            assert!(
                model.max_batch_size.unwrap() >= 1,
                "max_batch_size must be >= 1 for model {}",
                model.model_name
            );
            assert!(
                model.max_context_len.unwrap() >= 1,
                "max_context_len must be >= 1 for model {}",
                model.model_name
            );
            assert!(
                model.paragraph_len.unwrap() == 256,
                "paragraph_len must be exactly 256 for model {}",
                model.model_name
            );
        }
    }
}

pub static INFER_CFG: OnceCell<Arc<FinalInferConfig>> = OnceCell::new();
