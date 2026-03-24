use serde::{Deserialize, Serialize};
use sonic_rs::{JsonContainerTrait, JsonValueTrait, Value, from_str, json, to_string, to_string_pretty};
use tokio::sync::mpsc;

use crate::cores::queue::{
    GuidedDecodingConfig, QueueEvent, QueueFinishMeta, QueueOutput, QueueOutputCandidate,
    QueueOutputToken,
};
use crate::cores::queue::queue_worker::QueueSubmitRequest;
use crate::dtos::chat::completions::{
    ChatCompletionResp, ChatCompletionsChunkResponse, ChatCompletionsReq, Choices, ChunkChoice,
    ChunkToolCall, ChunkToolCallFunction, Content, Delta, Logprobs, Message, ResponseFormat, Tool,
    ToolCall, ToolChoice, TopLogprobs,
};
use crate::routes::AppState;
use crate::services::{
    ServiceError, ServiceResult, current_unix_seconds, next_id, select_queue,
    validate_chat_logprobs, validate_sampling_config,
};

pub struct ChatCompletionRun {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub stream_requested: bool,
    pub include_logprobs: bool,
    structured_plan: StructuredPlan,
    pub rx: mpsc::Receiver<QueueEvent>,
}

#[derive(Clone, Debug, Default)]
pub struct ChatStreamState {
    structured_buffer: String,
    emitted_structured_chunk: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChatStructuredResponseMode {
    PlainText,
    JsonText,
    ToolCall,
}

#[derive(Clone, Debug)]
struct StructuredPlan {
    prompt_preamble: Option<String>,
    guided_decoding_config: Option<GuidedDecodingConfig>,
    response_mode: ChatStructuredResponseMode,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ToolModelOutput {
    Message { content: String },
    ToolCalls { tool_calls: Vec<ToolModelCall> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ToolModelCall {
    name: String,
    arguments: Value,
}

impl ChatCompletionRun {
    pub async fn collect(mut self) -> ChatCompletionResp {
        let mut content = String::new();
        let mut tokens = Vec::new();
        let mut finish_meta = None;

        while let Some(event) = self.rx.recv().await {
            match event {
                QueueEvent::Delta(delta) => {
                    content.push_str(&delta.text);
                    if self.include_logprobs {
                        tokens.extend(delta.tokens);
                    }
                }
                QueueEvent::Done(meta) => {
                    finish_meta = Some(meta);
                    break;
                }
            }
        }

        let finish_meta = finish_meta.expect("queue stream closed without finish meta");
        match self.structured_plan.response_mode {
            ChatStructuredResponseMode::PlainText | ChatStructuredResponseMode::JsonText => {
                self.plain_response(content, tokens, &finish_meta)
            }
            ChatStructuredResponseMode::ToolCall => self.tool_call_response(&content, &finish_meta),
        }
    }

    pub fn new_stream_state(&self) -> ChatStreamState {
        ChatStreamState::default()
    }

    pub fn stream_role_chunk(&self) -> ChatCompletionsChunkResponse {
        ChatCompletionsChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        }
    }

    pub fn stream_chunks(
        &self,
        state: &mut ChatStreamState,
        delta: &QueueOutput,
    ) -> Vec<ChatCompletionsChunkResponse> {
        match self.structured_plan.response_mode {
            ChatStructuredResponseMode::PlainText | ChatStructuredResponseMode::JsonText => {
                vec![self.plain_stream_chunk(delta)]
            }
            ChatStructuredResponseMode::ToolCall => {
                state.structured_buffer.push_str(&delta.text);
                if state.emitted_structured_chunk {
                    return Vec::new();
                }

                match from_str::<ToolModelOutput>(&state.structured_buffer) {
                    Ok(parsed) => {
                        state.emitted_structured_chunk = true;
                        self.structured_stream_chunks(parsed)
                    }
                    Err(_) => Vec::new(),
                }
            }
        }
    }

    pub fn finish_chunks(
        &self,
        state: &mut ChatStreamState,
        finish_meta: &QueueFinishMeta,
    ) -> Vec<ChatCompletionsChunkResponse> {
        let mut chunks = Vec::new();

        let finish_reason = match self.structured_plan.response_mode {
            ChatStructuredResponseMode::PlainText | ChatStructuredResponseMode::JsonText => {
                finish_meta.reason.as_openai_str().to_string()
            }
            ChatStructuredResponseMode::ToolCall => {
                if !state.emitted_structured_chunk {
                    let parsed = parse_tool_model_output(&state.structured_buffer);
                    chunks.extend(self.structured_stream_chunks(parsed.clone()));
                    state.emitted_structured_chunk = true;
                }

                match parse_tool_model_output(&state.structured_buffer) {
                    ToolModelOutput::Message { .. } => {
                        finish_meta.reason.as_openai_str().to_string()
                    }
                    ToolModelOutput::ToolCalls { .. } => "tool_calls".to_string(),
                }
            }
        };

        chunks.push(ChatCompletionsChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
        });

        chunks
    }

    fn plain_response(
        &self,
        content: String,
        tokens: Vec<QueueOutputToken>,
        finish_meta: &QueueFinishMeta,
    ) -> ChatCompletionResp {
        ChatCompletionResp {
            id: self.id.clone(),
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![Choices {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(finish_meta.reason.as_openai_str().to_string()),
                logprobs: self.include_logprobs.then(|| build_chat_logprobs(&tokens)),
            }],
        }
    }

    fn tool_call_response(&self, text: &str, finish_meta: &QueueFinishMeta) -> ChatCompletionResp {
        let (message, finish_reason) = match parse_tool_model_output(text) {
            ToolModelOutput::Message { content } => (
                Message {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_meta.reason.as_openai_str().to_string(),
            ),
            ToolModelOutput::ToolCalls { tool_calls } => (
                Message {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(build_openai_tool_calls(&tool_calls)),
                    tool_call_id: None,
                },
                "tool_calls".to_string(),
            ),
        };

        ChatCompletionResp {
            id: self.id.clone(),
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![Choices {
                index: 0,
                message,
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
        }
    }

    fn plain_stream_chunk(&self, delta: &QueueOutput) -> ChatCompletionsChunkResponse {
        ChatCompletionsChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: (!delta.text.is_empty()).then_some(delta.text.clone()),
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: self
                    .include_logprobs
                    .then(|| build_chat_logprobs(&delta.tokens)),
            }],
        }
    }

    fn structured_stream_chunks(
        &self,
        parsed: ToolModelOutput,
    ) -> Vec<ChatCompletionsChunkResponse> {
        match parsed {
            ToolModelOutput::Message { content } => vec![ChatCompletionsChunkResponse {
                id: self.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: self.created,
                model: self.model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(content),
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
            }],
            ToolModelOutput::ToolCalls { tool_calls } => vec![ChatCompletionsChunkResponse {
                id: self.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: self.created,
                model: self.model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: Some(build_chunk_tool_calls(&tool_calls)),
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
            }],
        }
    }
}

pub async fn chat_completions(
    state: AppState,
    req: ChatCompletionsReq,
) -> ServiceResult<ChatCompletionRun> {
    if req.messages.is_empty() {
        return Err(ServiceError::bad_request("messages cannot be empty"));
    }

    let sampling = validate_sampling_config(
        req.temperature,
        req.top_k,
        req.top_p,
        req.max_tokens,
        req.presence_penalty,
        req.repetition_penalty,
        req.penalty_decay,
    )?;
    let handle = select_queue(&state, &req.model).await?;
    let token_logprobs_config = validate_chat_logprobs(
        req.logprobs,
        req.top_logprobs,
        req.candidate_token_texts.clone(),
        &handle.tokenizer,
    )?;
    let structured_plan = validate_chat_structured_output(&req, token_logprobs_config.is_some())?;
    let prompt = build_chat_prompt(
        &req.messages,
        structured_plan.prompt_preamble.as_deref(),
        structured_plan.response_mode,
    )?;
    let include_logprobs = token_logprobs_config.is_some();
    let rx = handle
        .submit(QueueSubmitRequest {
            prompt,
            sampling_config: sampling,
            token_logprobs_config,
            stop_suffixes: req.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
            guided_decoding_config: structured_plan.guided_decoding_config.clone(),
        })
        .await;

    Ok(ChatCompletionRun {
        id: next_id("chatcmpl"),
        created: current_unix_seconds(),
        model: req.model,
        stream_requested: req.stream.unwrap_or(false),
        include_logprobs,
        structured_plan,
        rx,
    })
}

fn validate_chat_structured_output(
    req: &ChatCompletionsReq,
    requested_token_logprobs: bool,
) -> ServiceResult<StructuredPlan> {
    let tools = req.tools.as_ref().filter(|tools| !tools.is_empty());

    if req.response_format.is_some() && tools.is_some() {
        return Err(ServiceError::bad_request(
            "response_format and tools cannot be used together yet",
        ));
    }

    if let Some(response_format) = req.response_format.as_ref() {
        if req.stop.is_some() {
            return Err(ServiceError::bad_request(
                "stop is not supported together with response_format",
            ));
        }
        return validate_response_format(response_format);
    }

    let Some(tools) = tools else {
        if req.tool_choice.is_some() {
            return Err(ServiceError::bad_request(
                "tool_choice requires tools to be provided",
            ));
        }
        if req.parallel_tool_calls.is_some() {
            return Err(ServiceError::bad_request(
                "parallel_tool_calls requires tools to be provided",
            ));
        }
        return Ok(StructuredPlan {
            prompt_preamble: None,
            guided_decoding_config: None,
            response_mode: ChatStructuredResponseMode::PlainText,
        });
    };

    if req.stop.is_some() {
        return Err(ServiceError::bad_request(
            "stop is not supported together with tools",
        ));
    }
    if requested_token_logprobs {
        return Err(ServiceError::bad_request(
            "logprobs are not supported together with tools yet",
        ));
    }

    let tools = validate_tools(tools)?;
    let tool_choice = validate_tool_choice(req.tool_choice.as_ref(), &tools)?;
    if tool_choice == ValidatedToolChoice::None {
        return Ok(StructuredPlan {
            prompt_preamble: None,
            guided_decoding_config: None,
            response_mode: ChatStructuredResponseMode::PlainText,
        });
    }

    let parallel_tool_calls = req.parallel_tool_calls.unwrap_or(false);
    let schema = build_tool_wrapper_schema(&tools, &tool_choice, parallel_tool_calls);
    let prompt_preamble = build_tool_prompt_preamble(&tools, &tool_choice, parallel_tool_calls);
    Ok(StructuredPlan {
        prompt_preamble: Some(prompt_preamble),
        guided_decoding_config: Some(GuidedDecodingConfig {
            schema_json: schema.to_string(),
            strict_mode: false,
        }),
        response_mode: ChatStructuredResponseMode::ToolCall,
    })
}

fn validate_response_format(response_format: &ResponseFormat) -> ServiceResult<StructuredPlan> {
    match response_format {
        ResponseFormat::Text => Ok(StructuredPlan {
            prompt_preamble: None,
            guided_decoding_config: None,
            response_mode: ChatStructuredResponseMode::PlainText,
        }),
        ResponseFormat::JsonObject => Ok(StructuredPlan {
            prompt_preamble: Some("Return only a valid JSON object.".to_string()),
            guided_decoding_config: Some(GuidedDecodingConfig {
                schema_json: json!({ "type": "object" }).to_string(),
                strict_mode: false,
            }),
            response_mode: ChatStructuredResponseMode::JsonText,
        }),
        ResponseFormat::JsonSchema { json_schema } => {
            let schema = json_schema.schema.as_ref().ok_or_else(|| {
                ServiceError::bad_request("response_format.json_schema.schema is required")
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

            Ok(StructuredPlan {
                prompt_preamble: Some(prompt),
                guided_decoding_config: Some(GuidedDecodingConfig {
                    schema_json: schema.to_string(),
                    strict_mode: json_schema.strict.unwrap_or(true),
                }),
                response_mode: ChatStructuredResponseMode::JsonText,
            })
        }
    }
}

fn validate_tools(tools: &[Tool]) -> ServiceResult<Vec<ValidatedToolDefinition>> {
    let mut validated = Vec::with_capacity(tools.len());
    let mut seen_names = Vec::<String>::with_capacity(tools.len());

    for tool in tools {
        if !tool.ty.eq_ignore_ascii_case("function") {
            return Err(ServiceError::bad_request(format!(
                "unsupported tool type: {}",
                tool.ty
            )));
        }

        let name = tool.function.name.trim();
        if name.is_empty() {
            return Err(ServiceError::bad_request(
                "tool.function.name cannot be empty",
            ));
        }
        if seen_names.iter().any(|existing| existing == name) {
            return Err(ServiceError::bad_request(format!(
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
    tool_choice: Option<&ToolChoice>,
    tools: &[ValidatedToolDefinition],
) -> ServiceResult<ValidatedToolChoice> {
    let Some(tool_choice) = tool_choice else {
        return Ok(ValidatedToolChoice::Auto);
    };

    match tool_choice {
        ToolChoice::Mode(mode) => {
            let mode = mode.trim();
            if mode.eq_ignore_ascii_case("none") {
                Ok(ValidatedToolChoice::None)
            } else if mode.eq_ignore_ascii_case("auto") {
                Ok(ValidatedToolChoice::Auto)
            } else if mode.eq_ignore_ascii_case("required") {
                Ok(ValidatedToolChoice::Required)
            } else {
                Err(ServiceError::bad_request(format!(
                    "unsupported tool_choice mode: {mode}"
                )))
            }
        }
        ToolChoice::Named(named) => {
            if !named.ty.eq_ignore_ascii_case("function") {
                return Err(ServiceError::bad_request(format!(
                    "unsupported tool_choice type: {}",
                    named.ty
                )));
            }
            let tool_name = named.function.name.trim();
            if tools.iter().any(|tool| tool.name == tool_name) {
                Ok(ValidatedToolChoice::Named(tool_name.to_string()))
            } else {
                Err(ServiceError::bad_request(format!(
                    "tool_choice references unknown function: {tool_name}"
                )))
            }
        }
    }
}

fn build_tool_wrapper_schema(
    tools: &[ValidatedToolDefinition],
    tool_choice: &ValidatedToolChoice,
    parallel_tool_calls: bool,
) -> Value {
    let message_variant = json!({
        "type": "object",
        "properties": {
            "type": { "const": "message" },
            "content": { "type": "string" }
        },
        "required": ["type", "content"],
        "additionalProperties": false
    });

    let tool_item_variants = tools
        .iter()
        .filter(|tool| match tool_choice {
            ValidatedToolChoice::Named(name) => tool.name == *name,
            _ => true,
        })
        .map(|tool| {
            json!({
                "type": "object",
                "properties": {
                    "name": { "const": tool.name },
                    "arguments": tool.parameters
                },
                "required": ["name", "arguments"],
                "additionalProperties": false
            })
        })
        .collect::<Vec<_>>();

    let tool_items = if tool_item_variants.len() == 1 {
        tool_item_variants[0].clone()
    } else {
        json!({ "oneOf": tool_item_variants })
    };

    let tool_calls_variant = if parallel_tool_calls {
        json!({
            "type": "object",
            "properties": {
                "type": { "const": "tool_calls" },
                "tool_calls": {
                    "type": "array",
                    "items": tool_items,
                    "minItems": 1
                }
            },
            "required": ["type", "tool_calls"],
            "additionalProperties": false
        })
    } else {
        json!({
            "type": "object",
            "properties": {
                "type": { "const": "tool_calls" },
                "tool_calls": {
                    "type": "array",
                    "items": tool_items,
                    "minItems": 1,
                    "maxItems": 1
                }
            },
            "required": ["type", "tool_calls"],
            "additionalProperties": false
        })
    };

    match tool_choice {
        ValidatedToolChoice::Auto => json!({ "oneOf": [message_variant, tool_calls_variant] }),
        ValidatedToolChoice::Required | ValidatedToolChoice::Named(_) => tool_calls_variant,
        ValidatedToolChoice::None => unreachable!("tool_choice=none should have returned early"),
    }
}

fn build_tool_prompt_preamble(
    tools: &[ValidatedToolDefinition],
    tool_choice: &ValidatedToolChoice,
    parallel_tool_calls: bool,
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
        }
        ValidatedToolChoice::Required => {
            lines.push(if parallel_tool_calls {
                "You must return one or more tool calls.".to_string()
            } else {
                "You must return exactly one tool call.".to_string()
            });
        }
        ValidatedToolChoice::Named(name) => {
            lines.push(if parallel_tool_calls {
                format!("You must call the tool `{name}` one or more times.")
            } else {
                format!("You must call the tool `{name}` exactly once.")
            });
        }
        ValidatedToolChoice::None => {}
    }

    lines.push(if parallel_tool_calls {
        "For tool calls, emit {\"type\":\"tool_calls\",\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}."
            .to_string()
    } else {
        "For a tool call, emit {\"type\":\"tool_calls\",\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}."
            .to_string()
    });

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

fn build_chat_prompt(
    messages: &[Message],
    prompt_preamble: Option<&str>,
    response_mode: ChatStructuredResponseMode,
) -> ServiceResult<String> {
    let mut prompt = String::new();

    if let Some(prompt_preamble) = prompt_preamble {
        prompt.push_str("System: ");
        prompt.push_str(prompt_preamble);
    }

    for message in messages {
        let rendered = render_prompt_message(message)?;
        if !prompt.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(&rendered);
    }

    if !prompt.is_empty() {
        prompt.push_str("\n\n");
    }
    match response_mode {
        ChatStructuredResponseMode::PlainText => prompt.push_str("Assistant: <think>\n"),
        ChatStructuredResponseMode::JsonText => prompt.push_str("Assistant: "),
        ChatStructuredResponseMode::ToolCall => {
            prompt.push_str("Assistant: <think>\n<|completions_of_cot|></think>```json\n");
        }
    }

    Ok(prompt)
}

fn render_prompt_message(message: &Message) -> ServiceResult<String> {
    let role = normalize_chat_role(&message.role)?;
    let has_tool_calls = message
        .tool_calls
        .as_ref()
        .is_some_and(|tool_calls| !tool_calls.is_empty());

    if has_tool_calls && role != "assistant" {
        return Err(ServiceError::bad_request(
            "tool_calls are only valid on assistant messages",
        ));
    }
    if message.tool_call_id.is_some() && role != "tool" {
        return Err(ServiceError::bad_request(
            "tool_call_id is only valid on tool messages",
        ));
    }

    match role {
        "user" => Ok(format!(
            "User: {}",
            message.content.as_deref().unwrap_or_default()
        )),
        "system" => Ok(format!(
            "System: {}",
            message.content.as_deref().unwrap_or_default()
        )),
        "tool" => Ok(format!(
            "User: {}",
            message.content.as_deref().unwrap_or_default()
        )),
        "assistant" if has_tool_calls => {
            if message
                .content
                .as_deref()
                .is_some_and(|content| !content.is_empty())
            {
                return Err(ServiceError::bad_request(
                    "assistant tool_calls messages cannot include content",
                ));
            }

            Ok(format!(
                "Assistant: <think>\n<|completions_of_cot|></think>```json\n{}\n```",
                render_tool_call_history_json(message.tool_calls.as_ref().expect("tool calls"))?
            ))
        }
        "assistant" => Ok(format!(
            "Assistant: {}",
            message.content.as_deref().unwrap_or_default()
        )),
        _ => unreachable!("role already normalized"),
    }
}

fn normalize_chat_role(role: &str) -> ServiceResult<&'static str> {
    match role.trim().to_ascii_lowercase().as_str() {
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "system" => Ok("system"),
        "tool" => Ok("tool"),
        _ => Err(ServiceError::bad_request(format!(
            "unknown chat role: {role}"
        ))),
    }
}

fn render_tool_call_history_json(tool_calls: &[ToolCall]) -> ServiceResult<String> {
    let tool_calls = tool_calls
        .iter()
        .enumerate()
        .map(|(index, tool_call)| {
            let arguments = from_str::<Value>(&tool_call.function.arguments).map_err(|err| {
                ServiceError::bad_request(format!(
                    "assistant tool_calls[{index}].function.arguments for function {:?} must be valid JSON: {err}",
                    tool_call.function.name
                ))
            })?;
            Ok(ToolModelCall {
                name: tool_call.function.name.clone(),
                arguments,
            })
        })
        .collect::<ServiceResult<Vec<_>>>()?;

    Ok(to_string(&ToolModelOutput::ToolCalls { tool_calls }).expect("tool call history json"))
}

fn parse_tool_model_output(text: &str) -> ToolModelOutput {
    let value = from_str::<Value>(text).expect("tool output must be valid structured json");
    let ty = value
        .get("type")
        .and_then(|ty| ty.as_str())
        .expect("tool output type");

    match ty {
        "message" => ToolModelOutput::Message {
            content: value
                .get("content")
                .and_then(|content| content.as_str())
                .expect("tool output message content")
                .to_string(),
        },
        "tool_calls" => ToolModelOutput::ToolCalls {
            tool_calls: value
                .get("tool_calls")
                .and_then(|tool_calls| tool_calls.as_array())
                .expect("tool output tool_calls array")
                .iter()
                .map(|tool_call| ToolModelCall {
                    name: tool_call
                        .get("name")
                        .and_then(|name| name.as_str())
                        .expect("tool call name")
                        .to_string(),
                    arguments: tool_call
                        .get("arguments")
                        .cloned()
                        .expect("tool call arguments"),
                })
                .collect(),
        },
        other => panic!("unsupported tool output type: {other}"),
    }
}

fn build_openai_tool_calls(tool_calls: &[ToolModelCall]) -> Vec<ToolCall> {
    tool_calls
        .iter()
        .map(|tool_call| ToolCall {
            id: next_id("call"),
            ty: "function".to_string(),
            function: crate::dtos::chat::completions::FunctionInContext {
                name: tool_call.name.clone(),
                arguments: to_string(&tool_call.arguments).expect("tool arguments json"),
            },
        })
        .collect()
}

fn build_chunk_tool_calls(tool_calls: &[ToolModelCall]) -> Vec<ChunkToolCall> {
    tool_calls
        .iter()
        .enumerate()
        .map(|(index, tool_call)| ChunkToolCall {
            index: index as u32,
            id: Some(next_id("call")),
            ty: Some("function".to_string()),
            function: Some(ChunkToolCallFunction {
                name: Some(tool_call.name.clone()),
                arguments: Some(to_string(&tool_call.arguments).expect("tool arguments json")),
            }),
        })
        .collect()
}

fn build_chat_logprobs(tokens: &[QueueOutputToken]) -> Logprobs {
    Logprobs {
        content: tokens.iter().map(build_chat_content).collect(),
    }
}

fn build_chat_content(token: &QueueOutputToken) -> Content {
    Content {
        token: token.token.clone(),
        bytes: token.bytes.clone(),
        logprob: token.logprob.unwrap_or(f32::NEG_INFINITY),
        top_logprobs: token
            .top_logprobs
            .iter()
            .map(build_chat_top_logprob)
            .collect(),
    }
}

fn build_chat_top_logprob(candidate: &QueueOutputCandidate) -> TopLogprobs {
    TopLogprobs {
        token: candidate.token.clone(),
        bytes: candidate.bytes.clone(),
        logprob: candidate.logprob,
    }
}
