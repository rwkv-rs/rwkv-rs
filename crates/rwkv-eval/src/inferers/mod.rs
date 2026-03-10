use async_openai::types::chat::{Prompt, StopConfiguration};
use async_openai::types::completions::CreateCompletionRequest as BaseCompletionRequest;
use serde::de::Unexpected::Option;
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
        sampling_config: &SamplingConfig,
    ) -> Self {
        Self {
            base: BaseCompletionRequest {
                model: model_name,
                prompt,
                max_tokens: Some(sampling_config.max_tokens),
                temperature: Some(sampling_config.temperature),
                top_p: Some(sampling_config.top_p),
                stop: Some(StopConfiguration::StringArray(sampling_config.stop_suffix.clone())),
                presence_penalty: Some(sampling_config.presence_penalty),
                frequency_penalty: Some(sampling_config.repetition_penalty),
                ..Default::default()
            },
            top_k: 0,
            penalty_decay: 0.0,
        }
    }
}


pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,

    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub penalty_decay: f32,

    pub max_tokens: u32,
    pub stop_suffix: Vec<String>,
}