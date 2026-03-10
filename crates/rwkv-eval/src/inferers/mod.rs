use crate::datasets::SamplingConfig;
use async_openai::types::chat::{Prompt, StopConfiguration};
use async_openai::types::completions::CreateCompletionRequest as BaseCompletionRequest;
use serde::Serialize;

#[derive(Serialize)]
pub struct CompletionRequest {
    #[serde(flatten)]
    base: BaseCompletionRequest,
    top_k: i32,
    penalty_decay: f32,
}

impl CompletionRequest {
    pub fn new(
        model_name: String,
        prompt: Prompt,
        stop_suffix: Vec<String>,
        max_tokens: u32,
        sampling_config: &SamplingConfig,
    ) -> Self {
        Self {
            base: BaseCompletionRequest {
                model: model_name,
                prompt,
                max_tokens: Some(max_tokens),
                temperature: Some(sampling_config.temperature),
                top_p: Some(sampling_config.top_p),
                stop: Some(StopConfiguration::StringArray(stop_suffix)),
                presence_penalty: Some(sampling_config.presence_penalty),
                frequency_penalty: Some(sampling_config.repetition_penalty),
                ..Default::default()
            },
            top_k: sampling_config.top_k,
            penalty_decay: sampling_config.penalty_decay,
        }
    }
}
