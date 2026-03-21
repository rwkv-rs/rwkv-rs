use crate::datasets::SamplingConfig;
use async_openai::types::chat::{Prompt, StopConfiguration};
use async_openai::types::completions::CreateCompletionRequest as BaseCompletionRequest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct CompletionRequest {
    #[serde(flatten)]
    base: BaseCompletionRequest,
    top_k: i32,
    penalty_decay: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_token_texts: Option<Vec<String>>,
}

impl CompletionRequest {
    pub fn new(
        model_name: String,
        prompt: Prompt,
        stop_suffix: Vec<String>,
        max_tokens: u32,
        sampling_config: &SamplingConfig,
        logprobs: Option<u8>,
        candidate_token_texts: Option<Vec<String>>,
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
                logprobs,
                ..Default::default()
            },
            top_k: sampling_config.top_k,
            penalty_decay: sampling_config.penalty_decay,
            candidate_token_texts,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionResponseChoice>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionResponseChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionLogprobs {
    pub tokens: Option<Vec<String>>,
    pub token_logprobs: Option<Vec<Option<f32>>>,
    pub top_logprobs: Option<Vec<BTreeMap<String, f32>>>,
    pub text_offset: Option<Vec<usize>>,
}
