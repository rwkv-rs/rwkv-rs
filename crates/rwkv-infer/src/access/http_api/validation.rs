use crate::access::http_api::{
    ChatCompletionRequest, ChatCompletionTool, ChatCompletionToolChoice, ResponseFormat,
};
use crate::inference_core::{ConstraintSpec, RequestedTokenLogprobsConfig, SamplingConfig};
use sonic_rs::{Value, json, to_string_pretty};

const DEFAULT_TEMPERATURE: f32 = 1.0;
const MIN_TEMPERATURE: f32 = 0.001;
const MAX_TEMPERATURE: f32 = 1000.0;
const DEFAULT_TOP_K: i32 = 0;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;
const DEFAULT_PRESENCE_PENALTY: f32 = 0.0;
const DEFAULT_REPETITION_PENALTY: f32 = 0.0;
const DEFAULT_PENALTY_DECAY: f32 = 1.0;
const MAX_COMPLETION_LOGPROBS: u8 = 5;
const MAX_CHAT_TOP_LOGPROBS: u8 = 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChatStructuredResponseMode {
    PlainText,
    JsonText,
    ToolCall,
}

#[derive(Clone, Debug)]
pub(crate) struct ChatStructuredOutputConfig {
    pub prompt_preamble: Option<String>,
    pub constraint: Option<ConstraintSpec>,
    pub response_mode: ChatStructuredResponseMode,
}

#[derive(Clone, Debug)]
struct ValidatedToolDefinition {
    name: String,
    description: Option<String>,
    parameters: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ValidatedToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_sampling_config(
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    max_new_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    penalty_decay: Option<f32>,
) -> crate::Result<SamplingConfig> {
    let temperature = temperature.unwrap_or(DEFAULT_TEMPERATURE);
    if !temperature.is_finite() || !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temperature) {
        return Err(crate::Error::bad_request(format!(
            "temperature must be finite and in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], got {temperature}"
        )));
    }

    let top_k = top_k.unwrap_or(DEFAULT_TOP_K);
    if top_k < 0 {
        return Err(crate::Error::bad_request(format!(
            "top_k must be >= 0, got {top_k}"
        )));
    }

    let top_p = top_p.unwrap_or(DEFAULT_TOP_P);
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(crate::Error::bad_request(format!(
            "top_p must be finite and in [0, 1], got {top_p}"
        )));
    }

    let max_new_tokens = max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    if max_new_tokens < 1 {
        return Err(crate::Error::bad_request(format!(
            "max_new_tokens must be >= 1, got {max_new_tokens}"
        )));
    }

    let presence_penalty = presence_penalty.unwrap_or(DEFAULT_PRESENCE_PENALTY);
    validate_finite("presence_penalty", presence_penalty)?;

    let repetition_penalty = repetition_penalty.unwrap_or(DEFAULT_REPETITION_PENALTY);
    validate_finite("repetition_penalty", repetition_penalty)?;

    let penalty_decay = penalty_decay.unwrap_or(DEFAULT_PENALTY_DECAY);
    validate_finite("penalty_decay", penalty_decay)?;

    Ok(SamplingConfig {
        temperature,
        top_k,
        top_p,
        max_new_tokens: max_new_tokens as usize,
        presence_penalty,
        repetition_penalty,
        penalty_decay,
    })
}

fn validate_finite(name: &str, value: f32) -> crate::Result<()> {
    if !value.is_finite() {
        return Err(crate::Error::bad_request(format!(
            "{name} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn normalize_candidate_token_texts(
    candidate_token_texts: Option<Vec<String>>,
) -> crate::Result<Option<Vec<String>>> {
    let Some(candidate_token_texts) = candidate_token_texts else {
        return Ok(None);
    };

    let mut deduped = Vec::with_capacity(candidate_token_texts.len());
    for token_text in candidate_token_texts {
        if token_text.is_empty() {
            return Err(crate::Error::bad_request(
                "candidate_token_texts cannot contain empty strings",
            ));
        }
        if !deduped.contains(&token_text) {
            deduped.push(token_text);
        }
    }

    if deduped.is_empty() {
        Ok(None)
    } else {
        Ok(Some(deduped))
    }
}

pub(crate) fn validate_completion_logprobs(
    logprobs: Option<u8>,
    candidate_token_texts: Option<Vec<String>>,
) -> crate::Result<Option<RequestedTokenLogprobsConfig>> {
    let candidate_token_texts = normalize_candidate_token_texts(candidate_token_texts)?;
    let Some(logprobs) = logprobs else {
        if candidate_token_texts.is_some() {
            return Err(crate::Error::bad_request(
                "candidate_token_texts requires logprobs to be set",
            ));
        }
        return Ok(None);
    };

    if logprobs > MAX_COMPLETION_LOGPROBS {
        return Err(crate::Error::bad_request(format!(
            "logprobs must be in [0, {MAX_COMPLETION_LOGPROBS}], got {logprobs}"
        )));
    }
    if candidate_token_texts.is_some() && logprobs == 0 {
        return Err(crate::Error::bad_request(
            "candidate_token_texts requires logprobs >= 1",
        ));
    }

    Ok(Some(RequestedTokenLogprobsConfig {
        top_logprobs: logprobs as usize,
        candidate_token_texts,
    }))
}

pub(crate) fn validate_chat_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    candidate_token_texts: Option<Vec<String>>,
) -> crate::Result<Option<RequestedTokenLogprobsConfig>> {
    let logprobs_enabled = logprobs.unwrap_or(false);
    let candidate_token_texts = normalize_candidate_token_texts(candidate_token_texts)?;

    if let Some(top_logprobs) = top_logprobs {
        if !logprobs_enabled {
            return Err(crate::Error::bad_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        if top_logprobs > MAX_CHAT_TOP_LOGPROBS {
            return Err(crate::Error::bad_request(format!(
                "top_logprobs must be in [0, {MAX_CHAT_TOP_LOGPROBS}], got {top_logprobs}"
            )));
        }
    }

    if !logprobs_enabled {
        if candidate_token_texts.is_some() {
            return Err(crate::Error::bad_request(
                "candidate_token_texts requires logprobs=true",
            ));
        }
        return Ok(None);
    }

    let top_logprobs = top_logprobs.unwrap_or(0);
    if candidate_token_texts.is_some() && top_logprobs == 0 {
        return Err(crate::Error::bad_request(
            "candidate_token_texts requires top_logprobs >= 1",
        ));
    }

    Ok(Some(RequestedTokenLogprobsConfig {
        top_logprobs: top_logprobs as usize,
        candidate_token_texts,
    }))
}

pub(crate) fn validate_chat_structured_output(
    req: &ChatCompletionRequest,
    requested_token_logprobs: &Option<RequestedTokenLogprobsConfig>,
) -> crate::Result<ChatStructuredOutputConfig> {
    let tools = req.tools.as_ref().filter(|tools| !tools.is_empty());
    if req.response_format.is_some() && tools.is_some() {
        return Err(crate::Error::bad_request(
            "response_format and tools cannot be used together yet",
        ));
    }

    if let Some(response_format) = req.response_format.as_ref() {
        if req.stop.is_some() {
            return Err(crate::Error::bad_request(
                "stop is not supported together with response_format",
            ));
        }
        return validate_response_format(response_format);
    }

    let Some(tools) = tools else {
        if req.tool_choice.is_some() {
            return Err(crate::Error::bad_request(
                "tool_choice requires tools to be provided",
            ));
        }
        return Ok(ChatStructuredOutputConfig {
            prompt_preamble: None,
            constraint: None,
            response_mode: ChatStructuredResponseMode::PlainText,
        });
    };

    if req.stop.is_some() {
        return Err(crate::Error::bad_request(
            "stop is not supported together with tools",
        ));
    }
    if req.parallel_tool_calls.unwrap_or(false) {
        return Err(crate::Error::bad_request(
            "parallel_tool_calls=true is not supported yet",
        ));
    }
    if requested_token_logprobs.is_some() {
        return Err(crate::Error::bad_request(
            "logprobs are not supported together with tools yet",
        ));
    }

    let tools = validate_tools(tools)?;
    let tool_choice = validate_tool_choice(req.tool_choice.as_ref(), &tools)?;
    if tool_choice == ValidatedToolChoice::None {
        return Ok(ChatStructuredOutputConfig {
            prompt_preamble: None,
            constraint: None,
            response_mode: ChatStructuredResponseMode::PlainText,
        });
    }

    let schema = build_tool_wrapper_schema(&tools, &tool_choice);
    let prompt_preamble = build_tool_prompt_preamble(&tools, &tool_choice);
    Ok(ChatStructuredOutputConfig {
        prompt_preamble: Some(prompt_preamble),
        constraint: Some(ConstraintSpec {
            schema_json: schema.to_string(),
            strict_mode: false,
        }),
        response_mode: ChatStructuredResponseMode::ToolCall,
    })
}

fn validate_response_format(
    response_format: &ResponseFormat,
) -> crate::Result<ChatStructuredOutputConfig> {
    match response_format {
        ResponseFormat::Text => Ok(ChatStructuredOutputConfig {
            prompt_preamble: None,
            constraint: None,
            response_mode: ChatStructuredResponseMode::PlainText,
        }),
        ResponseFormat::JsonObject => Ok(ChatStructuredOutputConfig {
            prompt_preamble: Some("Return only a valid JSON object.".to_string()),
            constraint: Some(ConstraintSpec {
                schema_json: json!({ "type": "object" }).to_string(),
                strict_mode: false,
            }),
            response_mode: ChatStructuredResponseMode::JsonText,
        }),
        ResponseFormat::JsonSchema { json_schema } => {
            let schema = json_schema.schema.as_ref().ok_or_else(|| {
                crate::Error::bad_request("response_format.json_schema.schema is required")
            })?;
            let mut prompt = format!(
                "Return only JSON that matches the provided schema `{}`.",
                json_schema.name
            );
            if let Some(description) = json_schema.description.as_ref() {
                prompt.push_str("\nSchema description: ");
                prompt.push_str(description);
            }
            prompt.push_str("\nJSON schema:\n");
            prompt.push_str(&to_string_pretty(schema).unwrap_or_else(|_| schema.to_string()));
            Ok(ChatStructuredOutputConfig {
                prompt_preamble: Some(prompt),
                constraint: Some(ConstraintSpec {
                    schema_json: schema.to_string(),
                    strict_mode: json_schema.strict.unwrap_or(true),
                }),
                response_mode: ChatStructuredResponseMode::JsonText,
            })
        }
    }
}

fn validate_tools(tools: &[ChatCompletionTool]) -> crate::Result<Vec<ValidatedToolDefinition>> {
    let mut validated = Vec::with_capacity(tools.len());
    let mut seen_names = Vec::<String>::with_capacity(tools.len());
    for tool in tools {
        if !tool.ty.eq_ignore_ascii_case("function") {
            return Err(crate::Error::bad_request(format!(
                "unsupported tool type: {}",
                tool.ty
            )));
        }
        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(crate::Error::bad_request(
                "tool.function.name cannot be empty",
            ));
        }
        if seen_names.iter().any(|existing| existing == name) {
            return Err(crate::Error::bad_request(format!(
                "duplicate tool name: {name}"
            )));
        }
        seen_names.push(name.to_string());
        validated.push(ValidatedToolDefinition {
            name: name.to_string(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone().unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })
            }),
        });
    }
    Ok(validated)
}

fn validate_tool_choice(
    tool_choice: Option<&ChatCompletionToolChoice>,
    tools: &[ValidatedToolDefinition],
) -> crate::Result<ValidatedToolChoice> {
    let Some(tool_choice) = tool_choice else {
        return Ok(ValidatedToolChoice::Auto);
    };

    match tool_choice {
        ChatCompletionToolChoice::Mode(mode) => {
            let mode = mode.trim();
            if mode.eq_ignore_ascii_case("none") {
                Ok(ValidatedToolChoice::None)
            } else if mode.eq_ignore_ascii_case("auto") {
                Ok(ValidatedToolChoice::Auto)
            } else if mode.eq_ignore_ascii_case("required") {
                Ok(ValidatedToolChoice::Required)
            } else {
                Err(crate::Error::bad_request(format!(
                    "unsupported tool_choice mode: {mode}"
                )))
            }
        }
        ChatCompletionToolChoice::Named(named) => {
            if !named.ty.eq_ignore_ascii_case("function") {
                return Err(crate::Error::bad_request(format!(
                    "unsupported tool_choice type: {}",
                    named.ty
                )));
            }
            let tool_name = named.function.name.trim();
            if tools.iter().any(|tool| tool.name == tool_name) {
                Ok(ValidatedToolChoice::Named(tool_name.to_string()))
            } else {
                Err(crate::Error::bad_request(format!(
                    "tool_choice references unknown function: {tool_name}"
                )))
            }
        }
    }
}

fn build_tool_wrapper_schema(
    tools: &[ValidatedToolDefinition],
    tool_choice: &ValidatedToolChoice,
) -> Value {
    let mut variants = Vec::<Value>::new();
    if *tool_choice == ValidatedToolChoice::Auto {
        variants.push(json!({
            "type": "object",
            "properties": {
                "type": { "const": "message" },
                "content": { "type": "string" }
            },
            "required": ["type", "content"],
            "additionalProperties": false
        }));
    }

    for tool in tools {
        if let ValidatedToolChoice::Named(name) = tool_choice
            && tool.name != *name
        {
            continue;
        }
        variants.push(json!({
            "type": "object",
            "properties": {
                "type": { "const": "tool_call" },
                "name": { "const": tool.name },
                "arguments": tool.parameters
            },
            "required": ["type", "name", "arguments"],
            "additionalProperties": false
        }));
    }

    json!({ "oneOf": variants })
}

fn build_tool_prompt_preamble(
    tools: &[ValidatedToolDefinition],
    tool_choice: &ValidatedToolChoice,
) -> String {
    let mut lines = vec![
        "You are using the OpenAI tool-calling interface.".to_string(),
        "Return only JSON matching the provided schema.".to_string(),
    ];

    match tool_choice {
        ValidatedToolChoice::Auto => {
            lines.push(
                "For a direct answer, emit {\"type\":\"message\",\"content\":\"...\"}.".to_string(),
            );
            lines.push(
                "To call a tool, emit {\"type\":\"tool_call\",\"name\":\"tool_name\",\"arguments\":{...}}.".to_string(),
            );
        }
        ValidatedToolChoice::Required => {
            lines.push("You must call exactly one tool.".to_string());
            lines.push(
                "Emit {\"type\":\"tool_call\",\"name\":\"tool_name\",\"arguments\":{...}}."
                    .to_string(),
            );
        }
        ValidatedToolChoice::Named(name) => {
            lines.push(format!("You must call the tool `{name}`."));
            lines.push(format!(
                "Emit {{\"type\":\"tool_call\",\"name\":\"{name}\",\"arguments\":{{...}}}}."
            ));
        }
        ValidatedToolChoice::None => {}
    }

    lines.push("Available tools:".to_string());
    for tool in tools {
        let mut line = format!("- {}", tool.name);
        if let Some(description) = tool.description.as_ref() {
            line.push_str(": ");
            line.push_str(description);
        }
        lines.push(line);
        lines.push(format!(
            "  parameters: {}",
            to_string_pretty(&tool.parameters).unwrap_or_else(|_| tool.parameters.to_string())
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use sonic_rs::from_str;
    use super::*;
    use crate::access::http_api::{
        ChatCompletionNamedToolChoice, ChatCompletionNamedToolChoiceFunction,
        ChatCompletionToolFunction, ChatMessage,
    };

    fn base_chat_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some("hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: None,
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            penalty_decay: None,
            stop: None,
            logprobs: None,
            top_logprobs: None,
            candidate_token_texts: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
        }
    }

    fn sample_tool(name: &str) -> ChatCompletionTool {
        ChatCompletionTool {
            ty: "function".to_string(),
            function: ChatCompletionToolFunction {
                name: name.to_string(),
                description: Some(format!("{name} description")),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"],
                    "additionalProperties": false
                })),
                strict: None,
            },
        }
    }

    #[test]
    fn validate_chat_structured_output_rejects_response_format_and_tools() {
        let mut req = base_chat_request();
        req.response_format = Some(ResponseFormat::JsonObject);
        req.tools = Some(vec![sample_tool("weather")]);

        let err = validate_chat_structured_output(&req, &None).unwrap_err();
        assert!(
            err.to_string()
                .contains("response_format and tools cannot be used together")
        );
    }

    #[test]
    fn validate_chat_structured_output_rejects_unknown_named_tool() {
        let mut req = base_chat_request();
        req.tools = Some(vec![sample_tool("weather")]);
        req.tool_choice = Some(ChatCompletionToolChoice::Named(
            ChatCompletionNamedToolChoice {
                ty: "function".to_string(),
                function: ChatCompletionNamedToolChoiceFunction {
                    name: "calendar".to_string(),
                },
            },
        ));

        let err = validate_chat_structured_output(&req, &None).unwrap_err();
        assert!(
            err.to_string()
                .contains("tool_choice references unknown function: calendar")
        );
    }

    #[test]
    fn validate_chat_structured_output_builds_tool_constraint_schema() {
        let mut req = base_chat_request();
        req.tools = Some(vec![sample_tool("weather")]);

        let config = validate_chat_structured_output(&req, &None).unwrap();
        assert_eq!(config.response_mode, ChatStructuredResponseMode::ToolCall);
        let constraint = config.constraint.expect("constraint");
        let schema: Value = from_str(&constraint.schema_json).unwrap();
        let schema_text = schema.to_string();
        assert!(schema_text.contains("\"oneOf\""));
        assert!(schema_text.contains("\"tool_call\""));
        assert!(schema_text.contains("\"weather\""));
    }
}
