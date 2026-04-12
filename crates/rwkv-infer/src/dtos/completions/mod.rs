use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::dtos::stop::StopField;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionsReq {
    pub model: String,
    pub prompt: String,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(alias = "frequency_penalty")]
    pub repetition_penalty: Option<f32>,
    pub penalty_decay: Option<f32>,
    pub stop: Option<StopField>,
    pub logprobs: Option<u8>,
    pub candidate_token_texts: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionsResp {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop_suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop_suffix_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<BTreeMap<String, f32>>,
    pub text_offset: Vec<usize>,
}
