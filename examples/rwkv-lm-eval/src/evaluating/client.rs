use async_openai::Client;
use async_openai::config::OpenAIConfig;
use rwkv_config::raw::eval::{ExtApiConfig, IntApiConfig};
use rwkv_eval::datasets::SamplingConfig;
use rwkv_eval::inferers::generate_text_completion;
use serde::{Deserialize, Serialize};
use sonic_rs::{JsonValueTrait, Value, json};
use std::collections::BTreeSet;
use std::env;

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
struct ChatFallbackRequest<'a> {
    model: &'a str,
    messages: Vec<ChatHealthMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
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

fn extract_chat_content(resp: ChatHealthResponse, model_name: &str) -> Result<String, String> {
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| format!("chat client `{model_name}` returned no choices"))?;
    choice
        .message
        .content
        .clone()
        .ok_or_else(|| format!("chat client `{model_name}` returned no content"))
}

fn validate_chat_probe_content(model_name: &str, content: &str) -> Result<(), String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err(format!("chat client `{model_name}` returned empty content"));
    }

    if let Ok(wire) = sonic_rs::from_str::<ChatHealthWire>(trimmed) {
        if wire.ok {
            return Ok(());
        }
        return Err(format!(
            "chat client `{model_name}` returned unhealthy probe response"
        ));
    }

    if let Ok(json) = sonic_rs::from_str::<Value>(trimmed) {
        if json
            .get("ok")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            return Ok(());
        }

        let status = json
            .get("status")
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_ascii_lowercase());
        if matches!(
            status.as_deref(),
            Some("available" | "healthy" | "ready" | "ok" | "operational" | "success")
        ) {
            return Ok(());
        }
    }

    if matches!(
        trimmed.to_ascii_uppercase().as_str(),
        "AVAILABLE" | "OK" | "HEALTHY" | "READY"
    ) {
        return Ok(());
    }

    Err(format!(
        "chat client `{model_name}` returned unsupported probe content: {trimmed:?}"
    ))
}

pub(crate) async fn check_completion_client(client: &Client<OpenAIConfig>, model_name: &str) {
    let _ = generate_text_completion(
        client,
        model_name,
        "ping",
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
    )
    .await
    .unwrap_or_else(|error| panic!("client `{model_name}` is unavailable: {error}"));
}

pub(crate) async fn check_chat_client(
    client: &Client<OpenAIConfig>,
    model_name: &str,
) -> Result<(), String> {
    let req = ChatHealthRequest {
        model: model_name,
        messages: vec![ChatHealthMessage {
            role: "user",
            content: "Return JSON confirming chat availability.",
        }],
        temperature: 0.001,
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

    let resp = client.chat().create_byot(&req).await;

    match resp {
        Ok(resp) => {
            let content = extract_chat_content(resp, model_name)?;
            if validate_chat_probe_content(model_name, &content).is_ok() {
                return Ok(());
            }
        }
        Err(error) => {
            let _ = error;
        }
    }

    let fallback_req = ChatFallbackRequest {
        model: model_name,
        messages: vec![ChatHealthMessage {
            role: "user",
            content: "Reply with exactly AVAILABLE.",
        }],
        temperature: 0.001,
        top_p: 1.0,
        max_completion_tokens: 16,
    };
    let fallback_resp: ChatHealthResponse = client
        .chat()
        .create_byot(&fallback_req)
        .await
        .map_err(|error| format!("chat client `{model_name}` is unavailable: {error}"))?;
    let content = extract_chat_content(fallback_resp, model_name)?;
    validate_chat_probe_content(model_name, &content)
}

fn parse_model_candidates(raw: &str) -> Vec<String> {
    raw.split([',', '\n', '\r'])
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn collect_model_candidates(
    configured_model: &str,
    role_candidates_env: &str,
    shared_candidates_env: &str,
) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();

    for env_name in [role_candidates_env, shared_candidates_env] {
        if let Ok(raw) = env::var(env_name) {
            for candidate in parse_model_candidates(&raw) {
                if seen.insert(candidate.clone()) {
                    ordered.push(candidate);
                }
            }
        }
    }

    let configured_model = configured_model.trim().to_string();
    if !configured_model.is_empty() && seen.insert(configured_model.clone()) {
        ordered.push(configured_model);
    }

    ordered
}

pub(crate) async fn resolve_chat_ext_api_config(
    role_label: &str,
    default_cfg: &ExtApiConfig,
    role_candidates_env: &str,
    shared_candidates_env: &str,
) -> ExtApiConfig {
    let client = build_client(&default_cfg.base_url, &default_cfg.api_key);
    let candidates = collect_model_candidates(
        &default_cfg.model,
        role_candidates_env,
        shared_candidates_env,
    );
    assert!(
        !candidates.is_empty(),
        "no candidate chat models configured for {role_label}"
    );

    let mut failures = Vec::new();
    for candidate in candidates {
        match check_chat_client(&client, &candidate).await {
            Ok(()) => {
                if candidate != default_cfg.model {
                    println!("selected fallback {role_label} model: {candidate}");
                } else {
                    println!("selected {role_label} model: {candidate}");
                }
                return ExtApiConfig {
                    base_url: default_cfg.base_url.clone(),
                    api_key: default_cfg.api_key.clone(),
                    model: candidate,
                };
            }
            Err(error) => failures.push(format!("{candidate}: {error}")),
        }
    }

    panic!(
        "no healthy chat model found for {role_label}; candidates failed: {}",
        failures.join(" | ")
    );
}

#[cfg(test)]
mod tests {
    use super::{collect_model_candidates, parse_model_candidates, validate_chat_probe_content};

    #[test]
    fn parse_candidates_accepts_commas_and_newlines() {
        let actual = parse_model_candidates("model-a, model-b\nmodel-c\r\nmodel-d");
        assert_eq!(actual, vec!["model-a", "model-b", "model-c", "model-d"]);
    }

    #[test]
    fn collect_candidates_deduplicates_and_keeps_order() {
        unsafe {
            std::env::set_var("RWKV_TEST_ROLE_MODELS", "model-b,model-a");
            std::env::set_var("RWKV_TEST_SHARED_MODELS", "model-a,model-c");
        }
        let actual = collect_model_candidates(
            "model-c",
            "RWKV_TEST_ROLE_MODELS",
            "RWKV_TEST_SHARED_MODELS",
        );
        assert_eq!(actual, vec!["model-b", "model-a", "model-c"]);
        unsafe {
            std::env::remove_var("RWKV_TEST_ROLE_MODELS");
            std::env::remove_var("RWKV_TEST_SHARED_MODELS");
        }
    }

    #[test]
    fn health_probe_accepts_strict_json() {
        assert!(validate_chat_probe_content("demo", r#"{"ok":true}"#).is_ok());
    }

    #[test]
    fn health_probe_accepts_status_json() {
        assert!(
            validate_chat_probe_content("demo", r#"{"status":"available","message":"ready"}"#)
                .is_ok()
        );
    }

    #[test]
    fn health_probe_accepts_plain_available() {
        assert!(validate_chat_probe_content("demo", "AVAILABLE").is_ok());
    }

    #[test]
    fn health_probe_rejects_empty_content() {
        assert!(validate_chat_probe_content("demo", "   ").is_err());
    }
}
