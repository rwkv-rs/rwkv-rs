use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::access::http_api::{
    ChatCompletionChunkChoice, ChatCompletionChunkDelta, ChatCompletionChunkResponse,
    ChatCompletionLogprobs, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionTokenLogprob, ChatCompletionTokenTopLogprob,
    ChatMessage, CompletionLogprobs, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, DeleteResponse, HealthResponse, ModelListResponse, ModelObject,
    ReloadModelsRequest, ReloadModelsResponse, ResponseIdRequest, ResponsesCreateRequest,
    ResponsesResource, StopField,
};
use crate::inference_core::InferenceSubmitResult as SubmitOutput;
use crate::inference_core::{
    EngineEvent, EntryId, FinishMetadata, InferenceOutput, StreamDelta, TokenLogprobsConfig,
};
use crate::model_pool::LoadedModelRegistry as RuntimeManager;
use crate::response_store::{
    BackgroundTaskState, CachedResponse, GLOBAL_BACKGROUND_TASKS, GLOBAL_RESPONSE_CACHE,
};

#[derive(Clone)]
pub struct HttpApiService {
    runtime_manager: Arc<RuntimeManager>,
}

impl HttpApiService {
    pub fn new(runtime_manager: Arc<RuntimeManager>) -> Self {
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
        let vocab_size = service.model_vocab_size(&req.model).ok_or_else(|| {
            crate::Error::bad_request(format!(
                "unknown model_name: {}. available: {:?}",
                req.model.as_str(),
                service.model_names()
            ))
        })?;
        let token_logprobs = crate::access::http_api::validation::validate_completion_logprobs(
            req.logprobs,
            req.candidate_token_ids,
            vocab_size,
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
                token_logprobs.clone(),
                Some(validate_ms),
            )
            .await?;
        let (_entry_id, rx) = expect_submit_receiver(submit)?;

        Ok(CompletionRun {
            id: format!("cmpl_{}", Uuid::new_v4().as_simple()),
            created: current_unix_seconds(),
            model: req.model,
            stream_requested,
            token_logprobs,
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
        let vocab_size = service.model_vocab_size(&req.model).ok_or_else(|| {
            crate::Error::bad_request(format!(
                "unknown model_name: {}. available: {:?}",
                req.model.as_str(),
                service.model_names()
            ))
        })?;
        let token_logprobs = crate::access::http_api::validation::validate_chat_logprobs(
            req.logprobs,
            req.top_logprobs,
            req.candidate_token_ids,
            vocab_size,
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
        let prompt = build_chat_prompt(&req.messages)?;

        let submit = service
            .submit_text_with_trace(
                &req.model,
                prompt,
                sampling,
                stop_suffixes,
                token_logprobs.clone(),
                Some(validate_ms),
            )
            .await?;
        let (_entry_id, rx) = expect_submit_receiver(submit)?;

        Ok(ChatCompletionRun {
            id: format!("chatcmpl_{}", Uuid::new_v4().as_simple()),
            created: current_unix_seconds(),
            model: req.model,
            stream_requested,
            token_logprobs,
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
        let patch = crate::model_pool::ModelsReloadPatch {
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
    pub token_logprobs: Option<TokenLogprobsConfig>,
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
        self.token_logprobs
            .as_ref()
            .map(|_| build_completion_logprobs(tokens, text_offset))
    }
}

#[derive(Debug)]
pub struct ChatCompletionRun {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub stream_requested: bool,
    pub token_logprobs: Option<TokenLogprobsConfig>,
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
                    content: delta.text.clone(),
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
        Ok(self.response(
            &StreamDelta {
                text: collected.text,
                tokens: collected.tokens,
            },
            Some(&collected.finish_meta),
        ))
    }

    fn chat_logprobs(&self, tokens: &[InferenceOutput]) -> Option<ChatCompletionLogprobs> {
        self.token_logprobs
            .as_ref()
            .map(|_| ChatCompletionLogprobs {
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

fn build_chat_prompt(messages: &[ChatMessage]) -> crate::Result<String> {
    let mut prompt = String::new();
    for msg in messages {
        let role = normalize_chat_role(&msg.role)
            .ok_or_else(|| crate::Error::bad_request(format!("unknown chat role: {}", msg.role)))?;
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(&msg.content);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Assistant: <think");
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
    } else {
        None
    }
}

pub type ApiService = HttpApiService;
