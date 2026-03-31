use std::collections::BTreeMap;

use async_openai::{Client, config::OpenAIConfig};
use async_openai::types::{
    chat::{Prompt, StopConfiguration},
    completions::CreateCompletionRequest as BaseCompletionRequest,
};
use serde::{Deserialize, Serialize};

use crate::cores::datasets::SamplingConfig;

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

#[derive(Clone, Debug, Serialize)]
struct FallbackChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<FallbackChatMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FallbackChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Clone, Debug, Deserialize)]
struct FallbackChatCompletionResponse {
    choices: Vec<FallbackChatChoice>,
}

#[derive(Clone, Debug, Deserialize)]
struct FallbackChatChoice {
    message: FallbackChatResponseMessage,
}

#[derive(Clone, Debug, Deserialize)]
struct FallbackChatResponseMessage {
    content: Option<String>,
}

fn prompt_as_chat_user_content(prompt: &str) -> &str {
    prompt
        .strip_prefix("User: ")
        .and_then(|rest| rest.strip_suffix("\n\nAssistant:"))
        .unwrap_or(prompt)
}

pub async fn generate_text_completion(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt: &str,
    stop_suffix: Vec<String>,
    max_tokens: u32,
    sampling_config: &SamplingConfig,
) -> Result<String, String> {
    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt.to_string().into(),
        stop_suffix.clone(),
        max_tokens,
        sampling_config,
        None,
        None,
    );

    let completion_resp: Result<CompletionResponse, _> =
        model_client.completions().create_byot(&req).await;
    let resp: CompletionResponse = match completion_resp {
        Ok(resp) => Ok(resp),
        Err(completion_err) => {
            let chat_req = FallbackChatCompletionRequest {
                model: model_name,
                messages: vec![FallbackChatMessage {
                    role: "user",
                    content: prompt_as_chat_user_content(prompt),
                }],
                temperature: sampling_config.temperature,
                top_p: sampling_config.top_p,
                max_tokens,
                stop: stop_suffix,
            };

            model_client
                .chat()
                .create_byot(&chat_req)
                .await
                .map_err(|chat_err| {
                    format!(
                        "completion request failed: {completion_err} | fallback chat failed: {chat_err}"
                    )
                })
                .and_then(|resp: FallbackChatCompletionResponse| {
                    resp.choices
                        .into_iter()
                        .next()
                        .and_then(|choice| choice.message.content)
                        .ok_or_else(|| {
                            "completion request failed and fallback chat returned no content"
                                .to_string()
                        })
                        .map(|content| CompletionResponse {
                            id: String::new(),
                            object: "chat.completion".to_string(),
                            created: 0,
                            model: model_name.to_string(),
                            choices: vec![CompletionResponseChoice {
                                text: content,
                                index: 0,
                                finish_reason: Some("stop".to_string()),
                                logprobs: None,
                            }],
                        })
                })
        }
    }?;

    resp.choices
        .into_iter()
        .next()
        .map(|choice| choice.text)
        .ok_or_else(|| format!("completion client `{model_name}` returned no choices"))
}
