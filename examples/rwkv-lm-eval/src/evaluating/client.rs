use async_openai::Client;
use async_openai::config::OpenAIConfig;
use rwkv_config::raw::eval::IntApiConfig;
use rwkv_eval::datasets::SamplingConfig;
use rwkv_eval::inferers::{CompletionRequest, CompletionResponse};
use serde::{Deserialize, Serialize};
use sonic_rs::{Value, json};

pub(crate) struct ClientWithConfig {
    pub api_cfg: IntApiConfig,
    pub client: Client<OpenAIConfig>,
}

pub(crate) fn build_client(base_url: &str, api_key: &str) -> Client<OpenAIConfig> {
    let config = OpenAIConfig::new()
        .with_api_key(api_key.to_string())
        .with_api_base(norm_api_url(base_url));

    Client::with_config(config)
}

fn norm_api_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    assert!(!base_url.is_empty(), "base_url cannot be empty");

    let base_url = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("http://{base_url}")
    };
    let base_url = base_url.trim_end_matches('/').to_string();

    if base_url.ends_with("/v1") {
        base_url
    } else {
        format!("{base_url}/v1")
    }
}

#[derive(Debug, Serialize)]
struct ChatHealthRequest<'a> {
    model: &'a str,
    messages: Vec<ChatHealthMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    response_format: ChatHealthResponseFormat,
}

#[derive(Debug, Serialize)]
struct ChatHealthMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatHealthResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: ChatHealthJsonSchema,
}

#[derive(Debug, Serialize)]
struct ChatHealthJsonSchema {
    description: Option<String>,
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
struct ChatHealthResponse {
    choices: Vec<ChatHealthChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatHealthChoice {
    message: ChatHealthResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatHealthResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatHealthWire {
    ok: bool,
}

pub(crate) async fn check_completion_client(client: &Client<OpenAIConfig>, model_name: &str) {
    let req = CompletionRequest::new(
        model_name.to_string(),
        "ping".into(),
        vec!["\n".to_string()],
        1,
        &SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            presence_penalty: 0.0,
            repetition_penalty: 0.0,
            penalty_decay: 1.0,
        },
        None,
        None,
    );

    let _: CompletionResponse = client
        .completions()
        .create_byot(&req)
        .await
        .unwrap_or_else(|error| panic!("client `{model_name}` is unavailable: {error}"));
}

pub(crate) async fn check_chat_client(client: &Client<OpenAIConfig>, model_name: &str) {
    let req = ChatHealthRequest {
        model: model_name,
        messages: vec![ChatHealthMessage {
            role: "user",
            content: "Return JSON confirming chat availability.",
        }],
        temperature: 0.0,
        top_p: 1.0,
        max_completion_tokens: 32,
        response_format: ChatHealthResponseFormat {
            kind: "json_schema",
            json_schema: ChatHealthJsonSchema {
                description: None,
                name: "chat_health".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "ok": { "type": "boolean" }
                    },
                    "required": ["ok"],
                    "additionalProperties": false
                }),
                strict: true,
            },
        },
    };

    let resp: ChatHealthResponse = client
        .chat()
        .create_byot(&req)
        .await
        .unwrap_or_else(|error| panic!("chat client `{model_name}` is unavailable: {error}"));
    let choice = resp
        .choices
        .first()
        .unwrap_or_else(|| panic!("chat client `{model_name}` returned no choices"));
    let content = choice
        .message
        .content
        .clone()
        .unwrap_or_else(|| panic!("chat client `{model_name}` returned no content"));
    let wire = sonic_rs::from_str::<ChatHealthWire>(&content)
        .unwrap_or_else(|err| panic!("chat client `{model_name}` returned invalid json: {err}; content={content:?}"));
    assert!(
        wire.ok,
        "chat client `{model_name}` returned unhealthy probe response"
    );
}
