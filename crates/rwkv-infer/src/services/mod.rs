pub mod admin;
pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod images;
pub mod models;
pub mod responses;

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use rwkv_config::raw::infer::GenerationConfig;
use rwkv_data::tokenizer::Tokenizer;
use uuid::Uuid;

use crate::{
    cores::{
        forward::{TokenIdLogprobsConfig, TokenTextLogprobsConfig, sampling::SamplingConfig},
        queue::queue_worker::QueueHandle,
    },
    dtos::errors::OpenAiErrorResponse,
    sonic_json::SonicJson,
};

pub type QueueMap = HashMap<String, Vec<QueueHandle>>;
pub type SharedQueueMap = Arc<RwLock<QueueMap>>;
pub type QueueMapBuilder =
    Arc<dyn Fn(&[GenerationConfig]) -> ServiceResult<QueueMap> + Send + Sync + 'static>;
pub type ServiceResult<T> = Result<T, ServiceError>;

const MAX_COMPLETION_LOGPROBS: u8 = 5;
const MAX_CHAT_TOP_LOGPROBS: u8 = 20;

#[derive(Clone, Debug)]
pub struct ServiceError {
    status: StatusCode,
    body: OpenAiErrorResponse,
}

impl ServiceError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: OpenAiErrorResponse::bad_request(message),
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            body: OpenAiErrorResponse::unauthorized(message),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: OpenAiErrorResponse::internal(message),
        }
    }

    pub fn not_supported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            body: OpenAiErrorResponse::not_supported(message),
        }
    }

    pub fn status_code(&self) -> StatusCode {
        self.status
    }

    pub fn body(&self) -> &OpenAiErrorResponse {
        &self.body
    }

    pub fn into_parts(self) -> (StatusCode, OpenAiErrorResponse) {
        (self.status, self.body)
    }
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        (self.status, SonicJson(self.body)).into_response()
    }
}

pub fn validate_sampling_config(
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    max_new_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    penalty_decay: Option<f32>,
) -> ServiceResult<SamplingConfig> {
    let temperature = temperature.unwrap_or(1.0);
    if !temperature.is_finite() || !(0.001..=1000.0).contains(&temperature) {
        return Err(ServiceError::bad_request(format!(
            "temperature must be finite and in [0.001, 1000], got {temperature}"
        )));
    }

    let top_k = top_k.unwrap_or(0);
    if top_k < 0 {
        return Err(ServiceError::bad_request(format!(
            "top_k must be >= 0, got {top_k}"
        )));
    }

    let top_p = top_p.unwrap_or(1.0);
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(ServiceError::bad_request(format!(
            "top_p must be finite and in [0, 1], got {top_p}"
        )));
    }

    let max_new_tokens = max_new_tokens.unwrap_or(256);
    if max_new_tokens < 1 {
        return Err(ServiceError::bad_request(format!(
            "max_tokens must be >= 1, got {max_new_tokens}"
        )));
    }

    let presence_penalty =
        finite_or_bad_request("presence_penalty", presence_penalty.unwrap_or(0.0))?;
    let repetition_penalty =
        finite_or_bad_request("repetition_penalty", repetition_penalty.unwrap_or(0.0))?;
    let penalty_decay = finite_or_bad_request("penalty_decay", penalty_decay.unwrap_or(1.0))?;

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

fn finite_or_bad_request(name: &str, value: f32) -> ServiceResult<f32> {
    if !value.is_finite() {
        return Err(ServiceError::bad_request(format!(
            "{name} must be finite, got {value}"
        )));
    }
    Ok(value)
}

pub fn model_queues<'a>(
    queues: &'a QueueMap,
    model_name: &str,
) -> ServiceResult<&'a [QueueHandle]> {
    if model_name.trim().is_empty() {
        return Err(ServiceError::bad_request(
            "model is required and cannot be empty",
        ));
    }

    let group = queues.get(model_name).ok_or_else(|| {
        ServiceError::bad_request(format!(
            "unknown model_name: {model_name}. available: {:?}",
            queues.keys().cloned().collect::<Vec<_>>()
        ))
    })?;

    Ok(group)
}

pub fn shared_queue_map(queues: QueueMap) -> SharedQueueMap {
    Arc::new(RwLock::new(queues))
}

pub fn select_queue(group: &[QueueHandle], model_name: &str) -> ServiceResult<QueueHandle> {
    if group.is_empty() {
        return Err(ServiceError::bad_request(format!(
            "model {model_name} has no available queues"
        )));
    }

    let handle = group
        .iter()
        .filter(|queue| queue.is_accepting())
        .min_by_key(|queue| (queue.load_score(), queue.device_id))
        .ok_or_else(|| {
            ServiceError::internal(format!("model {model_name} has no accepting queues"))
        })?;

    Ok(handle.clone())
}

pub fn select_model_queue(queues: &QueueMap, model_name: &str) -> ServiceResult<QueueHandle> {
    select_queue(model_queues(queues, model_name)?, model_name)
}

pub fn validate_completion_logprobs(
    logprobs: Option<u8>,
    candidate_token_texts: Option<Vec<String>>,
    tokenizer: &Tokenizer,
) -> ServiceResult<Option<TokenIdLogprobsConfig>> {
    let candidate_token_texts = normalize_candidate_token_texts(candidate_token_texts)?;
    let Some(logprobs) = logprobs else {
        if candidate_token_texts.is_some() {
            return Err(ServiceError::bad_request(
                "candidate_token_texts requires logprobs to be set",
            ));
        }
        return Ok(None);
    };

    if logprobs > MAX_COMPLETION_LOGPROBS {
        return Err(ServiceError::bad_request(format!(
            "logprobs must be in [0, {MAX_COMPLETION_LOGPROBS}], got {logprobs}"
        )));
    }
    if candidate_token_texts.is_some() && logprobs == 0 {
        return Err(ServiceError::bad_request(
            "candidate_token_texts requires logprobs >= 1",
        ));
    }

    resolve_token_logprobs_config(
        tokenizer,
        Some(TokenTextLogprobsConfig {
            top_logprobs: logprobs as usize,
            candidate_token_texts,
        }),
    )
}

pub fn validate_chat_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    candidate_token_texts: Option<Vec<String>>,
    tokenizer: &Tokenizer,
) -> ServiceResult<Option<TokenIdLogprobsConfig>> {
    let logprobs_enabled = logprobs.unwrap_or(false);
    let candidate_token_texts = normalize_candidate_token_texts(candidate_token_texts)?;

    if let Some(top_logprobs) = top_logprobs {
        if !logprobs_enabled {
            return Err(ServiceError::bad_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        if top_logprobs > MAX_CHAT_TOP_LOGPROBS {
            return Err(ServiceError::bad_request(format!(
                "top_logprobs must be in [0, {MAX_CHAT_TOP_LOGPROBS}], got {top_logprobs}"
            )));
        }
    }

    if !logprobs_enabled {
        if candidate_token_texts.is_some() {
            return Err(ServiceError::bad_request(
                "candidate_token_texts requires logprobs=true",
            ));
        }
        return Ok(None);
    }

    let top_logprobs = top_logprobs.unwrap_or(0);
    if candidate_token_texts.is_some() && top_logprobs == 0 {
        return Err(ServiceError::bad_request(
            "candidate_token_texts requires top_logprobs >= 1",
        ));
    }

    resolve_token_logprobs_config(
        tokenizer,
        Some(TokenTextLogprobsConfig {
            top_logprobs: top_logprobs as usize,
            candidate_token_texts,
        }),
    )
}

pub fn current_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub fn next_id(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4().simple())
}

fn normalize_candidate_token_texts(
    candidate_token_texts: Option<Vec<String>>,
) -> ServiceResult<Option<Vec<String>>> {
    let Some(candidate_token_texts) = candidate_token_texts else {
        return Ok(None);
    };

    let mut deduped = Vec::with_capacity(candidate_token_texts.len());
    for token_text in candidate_token_texts {
        if token_text.is_empty() {
            return Err(ServiceError::bad_request(
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

fn resolve_token_logprobs_config(
    tokenizer: &Tokenizer,
    requested: Option<TokenTextLogprobsConfig>,
) -> ServiceResult<Option<TokenIdLogprobsConfig>> {
    let Some(requested) = requested else {
        return Ok(None);
    };

    let candidate_token_ids = match requested.candidate_token_texts {
        Some(candidate_token_texts) => {
            let mut candidate_token_ids = Vec::with_capacity(candidate_token_texts.len());
            for token_text in candidate_token_texts {
                let encoded = tokenizer.encode(&token_text, false);
                if encoded.len() != 1 {
                    return Err(ServiceError::bad_request(format!(
                        "candidate_token_texts item {token_text:?} must tokenize to exactly one token, got {}",
                        encoded.len()
                    )));
                }

                let token_id = i32::from(encoded[0]);
                if !candidate_token_ids.contains(&token_id) {
                    candidate_token_ids.push(token_id);
                }
            }

            if candidate_token_ids.is_empty() {
                None
            } else {
                Some(candidate_token_ids)
            }
        }
        None => None,
    };

    Ok(Some(TokenIdLogprobsConfig {
        top_logprobs: requested.top_logprobs,
        candidate_token_ids,
    }))
}
