use rwkv_config::validated::infer::FinalInferConfig;

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_new_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_new_tokens: 256,
        }
    }
}

impl From<&FinalInferConfig> for SamplingConfig {
    fn from(cfg: &FinalInferConfig) -> Self {
        Self {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            max_new_tokens: cfg.max_new_tokens,
        }
    }
}
