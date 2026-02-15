use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RawInferConfig {
    // Paths
    pub model_config_path: String,
    pub weights_path: String,
    pub tokenizer_vocab_path: String,

    // HTTP
    pub http_bind_addr: Option<String>,
    pub request_body_limit_bytes: Option<usize>,
    pub sse_keep_alive_ms: Option<u64>,
    pub allowed_origins: Option<Vec<String>>,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,

    // Engine
    pub max_batch_size: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub max_context_length: Option<usize>,
    pub decode_first: Option<bool>,

    // Sampling
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub max_new_tokens: Option<usize>,

    // CubeCL DeviceId
    pub device_id_type: Option<u16>,
    pub device_id_index: Option<u32>,
}

impl RawInferConfig {
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            http_bind_addr: "0.0.0.0:8080".to_string(),
            request_body_limit_bytes: 50 * 1024 * 1024,
            sse_keep_alive_ms: 10_000,
            max_batch_size: 4,
            prefill_chunk_size: 256,
            max_context_length: 4096,
            decode_first: true,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_new_tokens: 256,
            device_id_type: 0u16,
            device_id_index: 0u32,
        );
    }
}
