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

#[cfg(test)]
mod tests {
    use super::CompletionsReq;

    #[test]
    fn parses_frequency_penalty_into_repetition_penalty() {
        let req: CompletionsReq = sonic_rs::from_str(
            r#"{
                "model":"demo",
                "prompt":"hi",
                "frequency_penalty":0.25
            }"#,
        )
        .expect("parse completion request");

        assert_eq!(req.repetition_penalty, Some(0.25));
    }
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
    pub logprobs: Option<Logprobs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<BTreeMap<String, f32>>,
    pub text_offset: Vec<usize>,
}
