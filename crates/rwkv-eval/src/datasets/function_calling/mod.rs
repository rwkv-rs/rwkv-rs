use crate::datasets::SamplingConfig;
use crate::inferers::{CompletionRequest, CompletionResponse};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sonic_rs::{Object as Map, Value, json};

pub mod tau_bench;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolRequestor {
    Assistant,
    User,
}

impl ToolRequestor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Assistant => "assistant",
            Self::User => "user",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub requestor: ToolRequestor,
    pub name: String,
    #[serde(default)]
    pub arguments: Map,
}

#[derive(Debug, Clone)]
pub struct FunctionCallingStep {
    pub cot: String,
    pub tool_call: FunctionCall,
    pub fc_output: String,
}

#[derive(Debug, Clone)]
pub enum FunctionCallingDecision {
    ToolCall(FunctionCall),
    FinalAnswer(String),
}

pub fn get_expected_context(
    system_prompt: &str,
    user_prompt: &str,
    steps: &[FunctionCallingStep],
) -> String {
    let mut text = format!("System: {system_prompt}\n\nUser: {user_prompt}\n\n");
    for step in steps {
        text.push_str("Assistant: <think>");
        text.push_str(&step.cot);
        text.push_str("</think>\n```json\n");
        text.push_str(
            &sonic_rs::to_string_pretty(&json!({
                "requestor": step.tool_call.requestor.as_str(),
                "name": step.tool_call.name,
                "arguments": step.tool_call.arguments,
            }))
            .unwrap(),
        );
        text.push_str("\n```\n\n");
        text.push_str("User: ");
        text.push_str(&step.fc_output);
        text.push_str("\n\n");
    }
    text.push_str("Assistant: <think><|completions_of_cot|>");
    text
}

pub fn build_turn_completion_prompt(cot_context: &str, cot: &str) -> String {
    cot_context.replace("<|completions_of_cot|>", cot) + "</think>\n"
}

pub async fn get_completion(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt: &str,
    sampling_config: &SamplingConfig,
    stop: Vec<String>,
    max_tokens: u32,
) -> String {
    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt.to_string().into(),
        stop,
        max_tokens,
        sampling_config,
        None,
        None,
    );
    let resp: CompletionResponse = model_client.completions().create_byot(&req).await.unwrap();
    resp.choices[0].text.clone()
}

pub fn parse_tool_call_or_final_answer(response: &str) -> Result<FunctionCallingDecision, String> {
    static JSON_BLOCK_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"(?s)```json\s*(?P<body>\{.*?\})\s*```").unwrap());

    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Err("model returned empty response".to_string());
    }

    let candidate = JSON_BLOCK_RE
        .captures(trimmed)
        .and_then(|caps| caps.name("body").map(|m| m.as_str().trim().to_string()))
        .or_else(|| {
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                Some(trimmed.to_string())
            } else {
                None
            }
        });

    if let Some(json_body) = candidate {
        let tool_call = sonic_rs::from_str::<FunctionCall>(&json_body)
            .map_err(|err| format!("failed to parse tool call json: {err}; json={json_body}"))?;
        return Ok(FunctionCallingDecision::ToolCall(tool_call));
    }

    Ok(FunctionCallingDecision::FinalAnswer(trimmed.to_string()))
}

pub fn render_fc_output(tool_call: &FunctionCall, outcome: Result<Value, String>) -> String {
    match outcome {
        Ok(output) => sonic_rs::to_string(&json!({
            "requestor": tool_call.requestor.as_str(),
            "name": tool_call.name,
            "ok": true,
            "output": output,
        }))
        .unwrap(),
        Err(error) => sonic_rs::to_string(&json!({
            "requestor": tool_call.requestor.as_str(),
            "name": tool_call.name,
            "ok": false,
            "error": error,
        }))
        .unwrap(),
    }
}
