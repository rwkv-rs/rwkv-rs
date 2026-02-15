use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::infer::RawInferConfig", cell = "INFER_CFG")]
pub struct FinalInferConfig {
    // Paths
    pub model_config_path: String,
    pub weights_path: String,
    pub tokenizer_vocab_path: String,

    // HTTP
    pub http_bind_addr: String,
    pub request_body_limit_bytes: usize,
    pub sse_keep_alive_ms: u64,
    pub allowed_origins: Option<Vec<String>>,

    #[serde(skip_serializing)]
    pub api_key: Option<String>,

    // Engine
    pub max_batch_size: usize,
    pub prefill_chunk_size: usize,
    pub max_context_length: usize,
    pub decode_first: bool,

    // Sampling
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_new_tokens: usize,

    // CubeCL DeviceId
    pub device_id_type: u16,
    pub device_id_index: u32,
}

impl FinalInferConfigBuilder {
    pub fn check(&self) {
        assert!(self.max_batch_size.unwrap() >= 1);
        assert!(self.prefill_chunk_size.unwrap() == 256);
        assert!(self.max_context_length.unwrap() >= 1);
        assert!(self.temperature.unwrap() > 0.0);
        assert!(self.top_p.unwrap() > 0.0 && self.top_p.unwrap() <= 1.0);
        assert!(self.max_new_tokens.unwrap() >= 1);
    }
}

pub static INFER_CFG: OnceCell<Arc<FinalInferConfig>> = OnceCell::new();
