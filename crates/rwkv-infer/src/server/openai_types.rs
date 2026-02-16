use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAiErrorResponse {
    pub error: OpenAiError,
}

impl OpenAiErrorResponse {
    pub fn not_supported(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "not_supported".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "invalid_request_error".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "authentication_error".to_string(),
                param: None,
                code: None,
            },
        }
    }
}

// === /v1/completions ===

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub penalty_decay: Option<f32>,
    pub stop: Option<StopField>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponseChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionResponseChoice>,
}

// === /v1/chat/completions ===

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub penalty_decay: Option<f32>,
    pub stop: Option<StopField>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    Single(String),
    Multiple(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::Single(s) => vec![s],
            StopField::Multiple(v) => v,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponseChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionResponseChoice>,
}

// === /v1/models ===

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}
