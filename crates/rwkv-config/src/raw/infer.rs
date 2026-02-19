use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct RawInferConfig {
    // HTTP
    pub http_bind_addr: Option<String>,
    pub request_body_limit_bytes: Option<usize>,
    pub sse_keep_alive_ms: Option<u64>,
    pub allowed_origins: Option<Vec<String>>,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,

    // Multi-model deployment
    pub models: Vec<GenerationConfig>,
}

impl RawInferConfig {
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            http_bind_addr: "0.0.0.0:8080".to_string(),
            request_body_limit_bytes: 50 * 1024 * 1024,
            sse_keep_alive_ms: 10_000u64,
        );

        for model in self.models.iter_mut() {
            model.fill_default();
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct GenerationConfig {
    pub model_name: String,
    #[serde(alias = "model_cfg_path")]
    pub model_cfg: String,
    pub weights_path: String,
    pub tokenizer_vocab_path: String,

    pub device_type: Option<u16>,
    pub device_ids: Vec<u32>,

    pub max_batch_size: Option<usize>,
    pub paragraph_len: Option<usize>,
    pub max_context_len: Option<usize>,
    pub decode_first: Option<bool>,
}

impl GenerationConfig {
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            device_type: 0u16,
            max_batch_size: 4usize,
            paragraph_len: 256usize,
            max_context_len: 4096usize,
            decode_first: true,
        );
    }
}
