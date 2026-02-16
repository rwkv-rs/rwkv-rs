#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_new_tokens: usize,

    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub penalty_decay: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_new_tokens: 256,
            presence_penalty: 0.0,
            repetition_penalty: 0.0,
            penalty_decay: 1.0,
        }
    }
}

impl SamplingConfig {
    pub fn penalties_enabled(&self) -> bool {
        self.presence_penalty != 0.0 || self.repetition_penalty != 0.0
    }
}

