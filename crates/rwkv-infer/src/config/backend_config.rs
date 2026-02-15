use std::net::SocketAddr;

use rwkv_config::validated::infer::FinalInferConfig;

#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub max_batch_size: usize,
    pub prefill_chunk_size: usize,
    pub max_context_length: usize,
    pub http_bind_addr: SocketAddr,
    pub request_body_limit_bytes: usize,
    pub sse_keep_alive_ms: u64,
    pub decode_first: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            prefill_chunk_size: 256,
            max_context_length: 4096,
            http_bind_addr: "0.0.0.0:8080".parse().expect("valid bind addr"),
            request_body_limit_bytes: 50 * 1024 * 1024,
            sse_keep_alive_ms: 10_000,
            decode_first: true,
        }
    }
}

impl From<&FinalInferConfig> for BackendConfig {
    fn from(cfg: &FinalInferConfig) -> Self {
        Self {
            max_batch_size: cfg.max_batch_size,
            prefill_chunk_size: cfg.prefill_chunk_size,
            max_context_length: cfg.max_context_length,
            http_bind_addr: cfg
                .http_bind_addr
                .parse()
                .expect("http_bind_addr must be SocketAddr"),
            request_body_limit_bytes: cfg.request_body_limit_bytes,
            sse_keep_alive_ms: cfg.sse_keep_alive_ms,
            decode_first: cfg.decode_first,
        }
    }
}
