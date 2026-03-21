use crate::datasets::SamplingConfig;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{Prompt, StopConfiguration};
use async_openai::types::completions::CreateCompletionRequest as BaseCompletionRequest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokio::time::{Duration, sleep};

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

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionFallbackRequest {
    model: String,
    messages: Vec<ChatCompletionFallbackMessage>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    presence_penalty: f32,
    frequency_penalty: f32,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionFallbackMessage {
    role: &'static str,
    content: String,
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
    #[serde(default)]
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
    #[serde(default)]
    pub message: Option<CompletionResponseMessage>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
}

impl CompletionResponseChoice {
    pub fn output_text(&self) -> &str {
        if !self.text.is_empty() {
            &self.text
        } else {
            self.message
                .as_ref()
                .and_then(|message| message.content.as_deref())
                .unwrap_or("")
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<BTreeMap<String, f32>>,
    pub text_offset: Vec<usize>,
}

pub async fn generate_text_completion(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt: &str,
    stop_suffix: Vec<String>,
    max_tokens: u32,
    sampling_config: &SamplingConfig,
) -> Result<String, String> {
    const MAX_ATTEMPTS: usize = 3;
    let mut failures = Vec::new();

    for attempt in 0..MAX_ATTEMPTS {
        let completion_req = CompletionRequest::new(
            model_name.to_string(),
            prompt.to_string().into(),
            stop_suffix.clone(),
            max_tokens,
            sampling_config,
            None,
            None,
        );

        match model_client
            .completions()
            .create_byot::<_, CompletionResponse>(&completion_req)
            .await
        {
            Ok(resp) => {
                let text = resp
                    .choices
                    .first()
                    .map(|choice| choice.output_text().to_string())
                    .unwrap_or_default();
                if !text.is_empty() {
                    return Ok(text);
                }
                failures.push(format!(
                    "attempt {} via completions returned empty content",
                    attempt + 1
                ));
            }
            Err(completion_err) => {
                failures.push(format!(
                    "attempt {} via completions failed: {completion_err}",
                    attempt + 1
                ));
            }
        }

        let chat_req = ChatCompletionFallbackRequest {
            model: model_name.to_string(),
            messages: vec![ChatCompletionFallbackMessage {
                role: "user",
                content: prompt.to_string(),
            }],
            temperature: sampling_config.temperature,
            top_p: sampling_config.top_p,
            max_completion_tokens: max_tokens,
            stop: stop_suffix.clone(),
            presence_penalty: sampling_config.presence_penalty,
            frequency_penalty: sampling_config.repetition_penalty,
        };
        match model_client
            .chat()
            .create_byot::<_, CompletionResponse>(&chat_req)
            .await
        {
            Ok(chat_resp) => {
                let text = chat_resp
                    .choices
                    .first()
                    .map(|choice| choice.output_text().to_string())
                    .unwrap_or_default();
                if !text.is_empty() {
                    return Ok(text);
                }
                failures.push(format!(
                    "attempt {} via fallback chat returned empty content",
                    attempt + 1
                ));
            }
            Err(chat_err) => {
                failures.push(format!(
                    "attempt {} via fallback chat failed: {chat_err}",
                    attempt + 1
                ));
            }
        }

        if attempt + 1 < MAX_ATTEMPTS {
            sleep(Duration::from_secs(1_u64 << attempt)).await;
        }
    }

    Err(failures.join(" | "))
}

#[cfg(test)]
mod tests {
    use super::CompletionResponse;

    #[test]
    fn completion_response_accepts_text_choices() {
        let raw = r#"{
          "id":"cmpl-1",
          "object":"text_completion",
          "created":1,
          "model":"demo",
          "choices":[{"text":"hello","index":0,"finish_reason":"stop","logprobs":null}]
        }"#;
        let resp: CompletionResponse = sonic_rs::from_str(raw).unwrap();
        assert_eq!(resp.choices[0].output_text(), "hello");
    }

    #[test]
    fn completion_response_accepts_chat_choices() {
        let raw = r#"{
          "id":"chatcmpl-1",
          "object":"chat.completion",
          "created":1,
          "model":"demo",
          "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}]
        }"#;
        let resp: CompletionResponse = sonic_rs::from_str(raw).unwrap();
        assert_eq!(resp.choices[0].output_text(), "hello");
    }
}
