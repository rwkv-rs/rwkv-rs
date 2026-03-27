use std::collections::BTreeMap;

use tokio::sync::mpsc;

use crate::{
    cores::queue::{
        QueueEvent,
        QueueFinishMeta,
        QueueOutput,
        QueueOutputToken,
        queue_worker::{QueueHandle, QueueSubmitRequest},
    },
    dtos::completions::{Choice, CompletionsReq, CompletionsResp, Logprobs},
    services::{
        ServiceResult,
        current_unix_seconds,
        next_id,
        validate_completion_logprobs,
        validate_sampling_config,
    },
};

pub struct CompletionRun {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub stream_requested: bool,
    pub include_logprobs: bool,
    pub rx: mpsc::Receiver<QueueEvent>,
}

impl CompletionRun {
    pub fn stream_chunk(&self, delta: &QueueOutput, text_offset: usize) -> CompletionsResp {
        CompletionsResp {
            id: self.id.clone(),
            object: "text_completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![Choice {
                text: delta.text.clone(),
                index: 0,
                finish_reason: None,
                logprobs: self
                    .include_logprobs
                    .then(|| build_completion_logprobs(&delta.tokens, text_offset)),
            }],
        }
    }

    pub async fn collect(mut self) -> CompletionsResp {
        let mut text = String::new();
        let mut tokens = Vec::new();
        let mut finish_meta = None;
        while let Some(event) = self.rx.recv().await {
            match event {
                QueueEvent::Delta(delta) => {
                    text.push_str(&delta.text);
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

        CompletionsResp {
            id: self.id,
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model,
            choices: vec![Choice {
                text,
                index: 0,
                finish_reason: Some(finish_meta.reason.as_openai_str().to_string()),
                logprobs: self
                    .include_logprobs
                    .then(|| build_completion_logprobs(&tokens, 0)),
            }],
        }
    }

    pub fn finish_chunk(&self, finish_meta: &QueueFinishMeta) -> CompletionsResp {
        CompletionsResp {
            id: self.id.clone(),
            object: "text_completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![Choice {
                text: String::new(),
                index: 0,
                finish_reason: Some(finish_meta.reason.as_openai_str().to_string()),
                logprobs: None,
            }],
        }
    }
}

pub async fn completions(handle: QueueHandle, req: CompletionsReq) -> ServiceResult<CompletionRun> {
    let sampling = validate_sampling_config(
        req.temperature,
        req.top_k,
        req.top_p,
        req.max_tokens,
        req.presence_penalty,
        req.repetition_penalty,
        req.penalty_decay,
    )?;
    let token_logprobs_config =
        validate_completion_logprobs(req.logprobs, req.candidate_token_texts, &handle.tokenizer)?;
    let include_logprobs = token_logprobs_config.is_some();
    let rx = handle
        .submit(QueueSubmitRequest {
            prompt: req.prompt,
            sampling_config: sampling,
            token_logprobs_config,
            stop_suffixes: req.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
            guided_decoding_config: None,
        })
        .await;

    Ok(CompletionRun {
        id: next_id("cmpl"),
        created: current_unix_seconds(),
        model: req.model,
        stream_requested: req.stream.unwrap_or(false),
        include_logprobs,
        rx,
    })
}

fn build_completion_logprobs(tokens: &[QueueOutputToken], text_offset: usize) -> Logprobs {
    let mut offsets = Vec::with_capacity(tokens.len());
    let mut current_offset = text_offset;
    for token in tokens {
        offsets.push(current_offset);
        current_offset += token.token.chars().count();
    }

    Logprobs {
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
