use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use sonic_rs::{Value, from_str, prelude::*, to_string};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{
    access::http_api::{
        ChatCompletionChunkChoice,
        ChatCompletionChunkDelta,
        ChatCompletionChunkResponse,
        ChatCompletionLogprobs,
        ChatCompletionMessageToolCall,
        ChatCompletionMessageToolCallFunction,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatCompletionTokenLogprob,
        ChatCompletionTokenTopLogprob,
        ChatMessage,
        CompletionLogprobs,
        CompletionRequest,
        CompletionResponse,
        CompletionResponseChoice,
        DeleteResponse,
        HealthResponse,
        ModelListResponse,
        ModelObject,
        ReloadModelsRequest,
        ReloadModelsResponse,
        ResponseIdRequest,
        ResponsesCreateRequest,
        ResponsesResource,
        StopField,
        validation::ChatStructuredResponseMode,
    },
    inference_core::{
        EngineEvent,
        EntryId,
        FinishMetadata,
        InferenceOutput,
        InferenceSubmitResult as SubmitOutput,
        StreamDelta,
    },
    model_pool::loaded_model_registry::{LoadedModelRegistry, ModelsReloadPatch},
    response_store::{
        BackgroundTaskState,
        CachedResponse,
        GLOBAL_BACKGROUND_TASKS,
        GLOBAL_RESPONSE_CACHE,
    },
};

#[derive(Clone)]
pub struct HttpApiService {
    runtime_manager: Arc<LoadedModelRegistry>,
}

impl HttpApiService {
    pub fn new(runtime_manager: Arc<LoadedModelRegistry>) -> Self {
        Self { runtime_manager }
    }

    pub fn models(&self) -> ModelListResponse {
        let data = self
            .runtime_manager
            .model_names()
            .into_iter()
            .map(|model_name| ModelObject {
                id: model_name,
                object: "model".to_string(),
                owned_by: "rwkv-rs".to_string(),
            })
            .collect();
        ModelListResponse {
            object: "list".to_string(),
            data,
        }
    }

    pub fn health(&self) -> HealthResponse {
        let _ = self;
        HealthResponse {
            status: "ok".to_string(),
        }
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "rwkv.infer.request.completions",
            skip_all,
            fields(model = %req.model, stream = req.stream.unwrap_or(false))
        )
    )]
    pub async fn completions(&self, req: CompletionRequest) -> crate::Result<CompletionRun> {
        if req.model.trim().is_empty() {
            return Err(crate::Error::bad_request(
                "model is required and cannot be empty",
            ));
        }

        let service = self.runtime_manager.current_request_router();
        if service.model_vocab_size(&req.model).is_none() {
            return Err(crate::Error::bad_request(format!(
                "unknown model_name: {}. available: {:?}",
                req.model.as_str(),
                service.model_names()
            )));
        }
        let requested_token_logprobs =
            crate::access::http_api::validation::validate_completion_logprobs(
                req.logprobs,
                req.candidate_token_texts,
            )?;
        let stream_requested = req.stream.unwrap_or(false);
        let validate_start = Instant::now();
        let sampling = crate::access::http_api::validation::validate_sampling_config(
            req.temperature,
            req.top_k,
            req.top_p,
            req.max_tokens,
            req.presence_penalty,
            req.repetition_penalty,
            req.penalty_decay,
        )?;
        let validate_ms = elapsed_ms(validate_start);
        #[cfg(feature = "trace")]
        tracing::info!(
            request_stage = "validate",
            validate_ms,
            "sampling validation completed"
        );
        let stop_suffixes = req.stop.map(StopField::into_vec).unwrap_or_default();

        let submit = service
            .submit_text_with_trace(
                &req.model,
                req.prompt,
                sampling,
                stop_suffixes,
                None,
                requested_token_logprobs.clone(),
                Some(validate_ms),
            )
            .await?;
        let (_entry_id, rx) = expect_submit_receiver(submit)?;

        Ok(CompletionRun {
            id: format!("cmpl_{}", Uuid::new_v4().as_simple()),
            created: current_unix_seconds(),
            model: req.model,
            stream_requested,
            include_logprobs: requested_token_logprobs.is_some(),
            rx,
        })
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "rwkv.infer.request.chat_completions",
            skip_all,
            fields(model = %req.model, stream = req.stream.unwrap_or(false))
        )
    )]
    pub async fn chat_completions(
        &self,
        req: ChatCompletionRequest,
    ) -> crate::Result<ChatCompletionRun> {
        if req.model.trim().is_empty() {
            return Err(crate::Error::bad_request(
                "model is required and cannot be empty",
            ));
        }

        let service = self.runtime_manager.current_request_router();
        if service.model_vocab_size(&req.model).is_none() {
            return Err(crate::Error::bad_request(format!(
                "unknown model_name: {}. available: {:?}",
                req.model.as_str(),
                service.model_names()
            )));
        }
        let requested_token_logprobs = crate::access::http_api::validation::validate_chat_logprobs(
            req.logprobs,
            req.top_logprobs,
            req.candidate_token_texts.clone(),
        )?;
        let structured_output =
            crate::access::http_api::validation::validate_chat_structured_output(
                &req,
                &requested_token_logprobs,
            )?;
        let stream_requested = req.stream.unwrap_or(false);
        if stream_requested
            && structured_output.response_mode == ChatStructuredResponseMode::ToolCall
        {
            return Err(crate::Error::bad_request(
                "stream=true is not supported together with tools yet",
            ));
        }
        let validate_start = Instant::now();
        let sampling = crate::access::http_api::validation::validate_sampling_config(
            req.temperature,
            req.top_k,
            req.top_p,
            req.max_tokens,
            req.presence_penalty,
            req.repetition_penalty,
            req.penalty_decay,
        )?;
        let validate_ms = elapsed_ms(validate_start);
        #[cfg(feature = "trace")]
        tracing::info!(
            request_stage = "validate",
            validate_ms,
            "sampling validation completed"
        );
        let stop_suffixes = req.stop.map(StopField::into_vec).unwrap_or_default();
        let prompt = build_chat_prompt(
            &req.messages,
            structured_output.prompt_preamble.as_deref(),
            structured_output.response_mode != ChatStructuredResponseMode::PlainText,
        )?;

        let submit = service
            .submit_text_with_trace(
                &req.model,
                prompt,
                sampling,
                stop_suffixes,
                structured_output.constraint,
                requested_token_logprobs.clone(),
                Some(validate_ms),
            )
            .await?;
        let (_entry_id, rx) = expect_submit_receiver(submit)?;

        Ok(ChatCompletionRun {
            id: format!("chatcmpl_{}", Uuid::new_v4().as_simple()),
            created: current_unix_seconds(),
            model: req.model,
            stream_requested,
            include_logprobs: requested_token_logprobs.is_some(),
            structured_output_mode: structured_output.response_mode,
            rx,
        })
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "rwkv.infer.request.responses_create",
            skip_all,
            fields(model = %req.model, background = req.background.unwrap_or(false))
        )
    )]
    pub async fn responses_create(
        &self,
        req: ResponsesCreateRequest,
    ) -> crate::Result<ResponsesResource> {
        if req.model.trim().is_empty() {
            return Err(crate::Error::bad_request(
                "model is required and cannot be empty",
            ));
        }

        let response_id = GLOBAL_RESPONSE_CACHE.next_response_id();
        let background = req.background.unwrap_or(false);
        let stop_suffixes = req.stop.map(StopField::into_vec).unwrap_or_default();
        let validate_start = Instant::now();
        let sampling = crate::access::http_api::validation::validate_sampling_config(
            req.temperature,
            req.top_k,
            req.top_p,
            req.max_output_tokens,
            req.presence_penalty,
            req.repetition_penalty,
            req.penalty_decay,
        )?;
        let validate_ms = elapsed_ms(validate_start);
        #[cfg(feature = "trace")]
        tracing::info!(
            request_stage = "validate",
            validate_ms,
            "sampling validation completed"
        );

        if background {
            let task = GLOBAL_BACKGROUND_TASKS.create(response_id.clone());
            let service = self.runtime_manager.current_request_router();
            let model_name = req.model;
            let input = req.input;
            let response_id_for_task = response_id.clone();
            tokio::spawn(async move {
                GLOBAL_BACKGROUND_TASKS.set_state(&task.task_id, BackgroundTaskState::InProgress);
                let submit = service
                    .submit_text_with_trace(
                        &model_name,
                        input,
                        sampling,
                        stop_suffixes,
                        None,
                        None,
                        Some(validate_ms),
                    )
                    .await;
                let mut rx = match submit {
                    Ok(SubmitOutput::Receiver { rx, .. }) => rx,
                    Ok(other) => {
                        GLOBAL_BACKGROUND_TASKS.fail(
                            &task.task_id,
                            format!("unexpected engine output: {other:?}"),
                        );
                        return;
                    }
                    Err(e) => {
                        GLOBAL_BACKGROUND_TASKS.fail(&task.task_id, e.to_string());
                        return;
                    }
                };

                match collect_stream_output(&mut rx).await {
                    Ok(collected) => {
                        GLOBAL_RESPONSE_CACHE.put(CachedResponse {
                            response_id: response_id_for_task,
                            output_text: collected.text,
                            output_token_ids: Vec::new(),
                        });
                        GLOBAL_BACKGROUND_TASKS
                            .set_state(&task.task_id, BackgroundTaskState::Completed);
                    }
                    Err(e) => {
                        GLOBAL_BACKGROUND_TASKS.fail(&task.task_id, e.to_string());
                    }
                }
            });

            return Ok(ResponsesResource {
                id: response_id,
                object: "response".to_string(),
                status: "queued".to_string(),
                output_text: None,
            });
        }

        let submit = self
            .runtime_manager
            .current_request_router()
            .submit_text_with_trace(
                &req.model,
                req.input,
                sampling,
                stop_suffixes,
                None,
                None,
                Some(validate_ms),
            )
            .await?;
        let (_entry_id, mut rx) = expect_submit_receiver(submit)?;
        let out = collect_stream_output(&mut rx).await?.text;

        GLOBAL_RESPONSE_CACHE.put(CachedResponse {
            response_id: response_id.clone(),
            output_text: out.clone(),
            output_token_ids: Vec::new(),
        });

        Ok(ResponsesResource {
            id: response_id,
            object: "response".to_string(),
            status: "completed".to_string(),
            output_text: Some(out),
        })
    }

    pub fn responses_get(&self, req: ResponseIdRequest) -> Option<ResponsesResource> {
        let _ = self;
        GLOBAL_RESPONSE_CACHE
            .get(&req.response_id)
            .map(|resp| ResponsesResource {
                id: resp.response_id,
                object: "response".to_string(),
                status: "completed".to_string(),
                output_text: Some(resp.output_text),
            })
    }

    pub fn responses_delete(&self, req: ResponseIdRequest) -> Option<DeleteResponse> {
        let _ = self;
        if GLOBAL_RESPONSE_CACHE.remove(&req.response_id) {
            Some(DeleteResponse {
                id: req.response_id,
                deleted: true,
            })
        } else {
            None
        }
    }

    pub fn responses_cancel(&self, _req: ResponseIdRequest) -> crate::Result<()> {
        let _ = self;
        Err(crate::Error::not_supported("cancel is not wired yet"))
    }

    pub async fn admin_models_reload(
        &self,
        req: ReloadModelsRequest,
    ) -> crate::Result<ReloadModelsResponse> {
        let patch = ModelsReloadPatch {
            upsert: req.upsert,
            remove_model_names: req.remove_model_names,
            dry_run: req.dry_run.unwrap_or(false),
        };

        let result = self.runtime_manager.reload_models(patch).await?;
        Ok(ReloadModelsResponse {
            changed_model_names: result.changed_model_names,
            rebuilt_model_names: result.rebuilt_model_names,
            removed_model_names: result.removed_model_names,
            active_model_names: result.active_model_names,
            dry_run: result.dry_run,
            message: result.message,
        })
    }

    pub fn embeddings(&self) -> crate::Result<()> {
        let _ = self;
        Err(crate::Error::not_supported(
            "/v1/embeddings not implemented yet",
        ))
    }

    pub fn images_generations(&self) -> crate::Result<()> {
        let _ = self;
        Err(crate::Error::not_supported(
            "/v1/images/generations not implemented yet",
        ))
    }

    pub fn audio_speech(&self) -> crate::Result<()> {
        let _ = self;
        Err(crate::Error::not_supported(
            "/v1/audio/speech not implemented yet",
        ))
    }
}

#[derive(Debug)]
pub struct CompletionRun {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub stream_requested: bool,
    pub include_logprobs: bool,
    pub rx: mpsc::Receiver<EngineEvent>,
}

impl CompletionRun {
    pub fn chunk(
        &self,
        delta: &StreamDelta,
        text_offset: usize,
        finish_meta: Option<&FinishMetadata>,
    ) -> CompletionResponse {
        CompletionResponse {
            id: self.id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![CompletionResponseChoice {
                text: delta.text.clone(),
                index: 0,
                finish_reason: finish_meta.map(|meta| meta.reason.as_openai_str().to_string()),
                logprobs: self.completion_logprobs(&delta.tokens, text_offset),
            }],
        }
    }

    pub async fn collect(mut self) -> crate::Result<CompletionResponse> {
        let collected = collect_stream_output(&mut self.rx).await?;
        Ok(self.chunk(
            &StreamDelta {
                text: collected.text,
                tokens: collected.tokens,
            },
            0,
            Some(&collected.finish_meta),
        ))
    }

    pub fn finish_chunk(&self, finish_meta: &FinishMetadata) -> CompletionResponse {
        CompletionResponse {
            id: self.id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![CompletionResponseChoice {
                text: String::new(),
                index: 0,
                finish_reason: Some(finish_meta.reason.as_openai_str().to_string()),
                logprobs: None,
            }],
        }
    }

    fn completion_logprobs(
        &self,
        tokens: &[InferenceOutput],
        text_offset: usize,
    ) -> Option<CompletionLogprobs> {
        self.include_logprobs
            .then(|| build_completion_logprobs(tokens, text_offset))
    }
}

#[derive(Debug)]
pub struct ChatCompletionRun {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub stream_requested: bool,
    pub include_logprobs: bool,
    structured_output_mode: ChatStructuredResponseMode,
    pub rx: mpsc::Receiver<EngineEvent>,
}

impl ChatCompletionRun {
    pub fn response(
        &self,
        delta: &StreamDelta,
        finish_meta: Option<&FinishMetadata>,
    ) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: self.id.clone(),
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatCompletionResponseChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(delta.text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: finish_meta.map(|meta| meta.reason.as_openai_str().to_string()),
                logprobs: self.chat_logprobs(&delta.tokens),
            }],
        }
    }

    pub fn stream_role_chunk(&self) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        }
    }

    pub fn stream_chunk(
        &self,
        delta: &StreamDelta,
        finish_meta: Option<&FinishMetadata>,
    ) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: (!delta.text.is_empty()).then_some(delta.text.clone()),
                    tool_calls: None,
                },
                finish_reason: finish_meta.map(|meta| meta.reason.as_openai_str().to_string()),
                logprobs: self.chat_logprobs(&delta.tokens),
            }],
        }
    }

    pub fn finish_chunk(&self, finish_meta: &FinishMetadata) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta::default(),
                finish_reason: Some(finish_meta.reason.as_openai_str().to_string()),
                logprobs: None,
            }],
        }
    }

    pub async fn collect(mut self) -> crate::Result<ChatCompletionResponse> {
        let collected = collect_stream_output(&mut self.rx).await?;
        match self.structured_output_mode {
            ChatStructuredResponseMode::PlainText | ChatStructuredResponseMode::JsonText => {
                Ok(self.response(
                    &StreamDelta {
                        text: collected.text,
                        tokens: collected.tokens,
                    },
                    Some(&collected.finish_meta),
                ))
            }
            ChatStructuredResponseMode::ToolCall => {
                Ok(self.tool_call_response(&collected.text, &collected.finish_meta))
            }
        }
    }

    fn chat_logprobs(&self, tokens: &[InferenceOutput]) -> Option<ChatCompletionLogprobs> {
        self.include_logprobs.then(|| ChatCompletionLogprobs {
            content: tokens
                .iter()
                .map(|token| ChatCompletionTokenLogprob {
                    token: token.token.clone(),
                    bytes: token.bytes.clone(),
                    logprob: token.logprob.unwrap_or(f32::NEG_INFINITY),
                    top_logprobs: token
                        .top_logprobs
                        .iter()
                        .map(|candidate| ChatCompletionTokenTopLogprob {
                            token: candidate.token.clone(),
                            bytes: candidate.bytes.clone(),
                            logprob: candidate.logprob,
                        })
                        .collect(),
                })
                .collect(),
        })
    }

    fn tool_call_response(
        &self,
        text: &str,
        finish_meta: &FinishMetadata,
    ) -> ChatCompletionResponse {
        let parsed = from_str::<Value>(text)
            .ok()
            .and_then(parse_structured_assistant_output);
        let (message, finish_reason) = match parsed {
            Some(StructuredAssistantOutput::Message { content }) => (
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_meta.reason.as_openai_str().to_string(),
            ),
            Some(StructuredAssistantOutput::ToolCall { name, arguments }) => (
                ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ChatCompletionMessageToolCall {
                        id: format!("call_{}", Uuid::new_v4().as_simple()),
                        ty: "function".to_string(),
                        function: ChatCompletionMessageToolCallFunction {
                            name,
                            arguments: arguments.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                "tool_calls".to_string(),
            ),
            None => (
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(text.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_meta.reason.as_openai_str().to_string(),
            ),
        };

        ChatCompletionResponse {
            id: self.id.clone(),
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatCompletionResponseChoice {
                index: 0,
                message,
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
        }
    }
}

enum StructuredAssistantOutput {
    Message { content: String },
    ToolCall { name: String, arguments: Value },
}

fn parse_structured_assistant_output(value: Value) -> Option<StructuredAssistantOutput> {
    let ty = value.get("type")?.as_str()?;
    match ty {
        "message" => Some(StructuredAssistantOutput::Message {
            content: value.get("content")?.as_str()?.to_string(),
        }),
        "tool_call" => Some(StructuredAssistantOutput::ToolCall {
            name: value.get("name")?.as_str()?.to_string(),
            arguments: value.get("arguments")?.clone(),
        }),
        _ => None,
    }
}

pub struct CollectedStream {
    pub text: String,
    pub tokens: Vec<InferenceOutput>,
    pub finish_meta: FinishMetadata,
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(name = "rwkv.infer.request.collect_stream", skip(rx))
)]
pub async fn collect_stream_output(
    rx: &mut mpsc::Receiver<EngineEvent>,
) -> crate::Result<CollectedStream> {
    let mut out = String::new();
    let mut tokens = Vec::new();
    let mut finish_meta = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            EngineEvent::Output(delta) => {
                #[cfg(feature = "trace")]
                tracing::trace!(
                    chars = delta.text.chars().count(),
                    tokens = delta.tokens.len(),
                    "collect stream chunk"
                );
                out.push_str(&delta.text);
                tokens.extend(delta.tokens);
            }
            EngineEvent::Done(meta) => {
                #[cfg(feature = "trace")]
                tracing::info!(
                    finish_reason = meta.reason.as_openai_str(),
                    generated_tokens = meta.generated_tokens,
                    "collect stream done"
                );
                finish_meta = Some(meta);
                break;
            }
            EngineEvent::Error(msg) => {
                #[cfg(feature = "trace")]
                tracing::error!(error = %msg, "collect stream failed");
                return Err(crate::Error::internal(msg));
            }
        }
    }
    let finish_meta = finish_meta.ok_or_else(|| {
        crate::Error::internal("engine stream closed without final finish reason")
    })?;
    Ok(CollectedStream {
        text: out,
        tokens,
        finish_meta,
    })
}

fn build_completion_logprobs(tokens: &[InferenceOutput], text_offset: usize) -> CompletionLogprobs {
    let mut offsets = Vec::with_capacity(tokens.len());
    let mut current_offset = text_offset;
    for token in tokens {
        offsets.push(current_offset);
        current_offset += token.token.chars().count();
    }

    CompletionLogprobs {
        tokens: tokens.iter().map(|token| token.token.clone()).collect(),
        token_logprobs: tokens.iter().map(|token| token.logprob).collect(),
        top_logprobs: tokens
            .iter()
            .map(|token| {
                let mut map = BTreeMap::new();
                for candidate in &token.top_logprobs {
                    map.insert(candidate.token.clone(), candidate.logprob);
                }
                map
            })
            .collect(),
        text_offset: offsets,
    }
}

fn expect_submit_receiver(
    submit: SubmitOutput,
) -> crate::Result<(EntryId, mpsc::Receiver<EngineEvent>)> {
    match submit {
        SubmitOutput::Receiver { entry_id, rx } => Ok((entry_id, rx)),
        SubmitOutput::Error { message, .. } => Err(crate::Error::bad_request(message)),
    }
}

fn current_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

fn build_chat_prompt(
    messages: &[ChatMessage],
    prompt_preamble: Option<&str>,
    structured_output: bool,
) -> crate::Result<String> {
    let mut prompt = String::new();
    if let Some(prompt_preamble) = prompt_preamble {
        prompt.push_str("System: ");
        prompt.push_str(prompt_preamble);
        prompt.push_str("\n\n");
    }
    for msg in messages {
        let role = normalize_chat_role(&msg.role)
            .ok_or_else(|| crate::Error::bad_request(format!("unknown chat role: {}", msg.role)))?;
        prompt.push_str(role);
        prompt.push_str(": ");
        if let Some(content) = msg.content.as_deref() {
            prompt.push_str(content);
        }
        if let Some(tool_calls) = msg.tool_calls.as_ref() {
            if !tool_calls.is_empty() {
                if msg
                    .content
                    .as_deref()
                    .is_some_and(|content| !content.is_empty())
                {
                    prompt.push('\n');
                }
                prompt.push_str(
                    &to_string(tool_calls).unwrap_or_else(|_| "<tool_calls>".to_string()),
                );
            }
        }
        prompt.push_str("\n\n");
    }
    prompt.push_str("Assistant: ");
    if !structured_output {
        prompt.push_str("<think");
    }
    Ok(prompt)
}

fn normalize_chat_role(role: &str) -> Option<&'static str> {
    let role = role.trim();
    if role.eq_ignore_ascii_case("user") {
        Some("User")
    } else if role.eq_ignore_ascii_case("assistant") {
        Some("Assistant")
    } else if role.eq_ignore_ascii_case("system") {
        Some("System")
    } else if role.eq_ignore_ascii_case("tool") {
        Some("Tool")
    } else {
        None
    }
}

pub type ApiService = HttpApiService;

#[cfg(test)]
mod tests {
    use sonic_rs::{from_str, json};

    use super::*;
    use crate::inference_core::{FinishReason, TimingBreakdownMs};

    fn finish_metadata(reason: FinishReason) -> FinishMetadata {
        FinishMetadata {
            reason,
            matched_stop_suffix: None,
            matched_stop_suffix_index: None,
            max_new_tokens: 16,
            generated_tokens: 1,
            timings_ms: Some(TimingBreakdownMs::default()),
        }
    }

    fn make_chat_run(mode: ChatStructuredResponseMode) -> ChatCompletionRun {
        let (_tx, rx) = mpsc::channel(1);
        ChatCompletionRun {
            id: "chatcmpl_test".to_string(),
            created: 0,
            model: "test-model".to_string(),
            stream_requested: false,
            include_logprobs: false,
            structured_output_mode: mode,
            rx,
        }
    }

    #[test]
    fn tool_call_response_maps_wrapper_json() {
        let run = make_chat_run(ChatStructuredResponseMode::ToolCall);
        let response = run.tool_call_response(
            r#"{"type":"tool_call","name":"weather","arguments":{"city":"Shanghai","unit":"c"}}"#,
            &finish_metadata(FinishReason::Stop),
        );

        let choice = &response.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert!(choice.message.content.is_none());
        let tool_calls = choice.message.tool_calls.as_ref().expect("tool calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].ty, "function");
        assert_eq!(tool_calls[0].function.name, "weather");
        assert_eq!(
            from_str::<Value>(&tool_calls[0].function.arguments).unwrap(),
            json!({"city": "Shanghai", "unit": "c"})
        );
    }

    #[test]
    fn tool_call_response_falls_back_to_plain_text_for_invalid_json() {
        let run = make_chat_run(ChatStructuredResponseMode::ToolCall);
        let response =
            run.tool_call_response("not valid json", &finish_metadata(FinishReason::Stop));

        let choice = &response.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
        assert_eq!(choice.message.content.as_deref(), Some("not valid json"));
        assert!(choice.message.tool_calls.is_none());
    }
}
