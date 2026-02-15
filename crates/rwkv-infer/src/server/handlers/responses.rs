use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::auth::check_api_key;
use crate::config::SamplingConfig;
use crate::engine::SubmitOutput;
use crate::server::SharedRwkvInferState;
use crate::server::openai_types::OpenAiErrorResponse;
use crate::storage::{GLOBAL_BACKGROUND_TASKS, GLOBAL_RESPONSE_CACHE};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesCreateRequest {
    pub model: Option<String>,
    pub input: String,
    pub background: Option<bool>,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesResource {
    pub id: String,
    pub object: String,
    pub status: String,
    pub output_text: Option<String>,
}

pub async fn responses_create(
    headers: HeaderMap,
    State(state): State<SharedRwkvInferState>,
    Json(req): Json<ResponsesCreateRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &state.auth) {
        return resp;
    }

    let response_id = GLOBAL_RESPONSE_CACHE.next_response_id();
    let background = req.background.unwrap_or(false);

    let sampling = SamplingConfig {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: 0,
        top_p: req.top_p.unwrap_or(1.0),
        max_new_tokens: req.max_output_tokens.unwrap_or(256) as usize,
    };

    if background {
        let task = GLOBAL_BACKGROUND_TASKS.create(response_id.clone());
        let engine = state.engine.clone();
        let response_id_for_task = response_id.clone();
        tokio::spawn(async move {
            GLOBAL_BACKGROUND_TASKS.set_state(
                &task.task_id,
                crate::storage::BackgroundTaskState::InProgress,
            );
            let submit = engine.submit_text(req.input, sampling, true).await;
            let mut rx = match submit {
                Ok(SubmitOutput::Stream { rx, .. }) => rx,
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

            let mut out = String::new();
            while let Some(ev) = rx.recv().await {
                match ev {
                    crate::types::EngineEvent::Text(t) => out.push_str(&t),
                    crate::types::EngineEvent::Done => break,
                    crate::types::EngineEvent::Error(msg) => {
                        GLOBAL_BACKGROUND_TASKS.fail(&task.task_id, msg);
                        return;
                    }
                }
            }

            GLOBAL_RESPONSE_CACHE.put(crate::storage::CachedResponse {
                response_id: response_id_for_task.clone(),
                output_text: out,
                output_token_ids: Vec::new(),
            });
            GLOBAL_BACKGROUND_TASKS.set_state(
                &task.task_id,
                crate::storage::BackgroundTaskState::Completed,
            );
        });

        return (
            StatusCode::OK,
            Json(ResponsesResource {
                id: response_id,
                object: "response".to_string(),
                status: "queued".to_string(),
                output_text: None,
            }),
        )
            .into_response();
    }

    // Foreground: run immediately (collect).
    let submit = state.engine.submit_text(req.input, sampling, true).await;
    let mut rx = match submit {
        Ok(SubmitOutput::Stream { rx, .. }) => rx,
        Ok(other) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::bad_request(format!(
                    "unexpected engine output: {other:?}"
                ))),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::bad_request(e.to_string())),
            )
                .into_response();
        }
    };

    let mut out = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            crate::types::EngineEvent::Text(t) => out.push_str(&t),
            crate::types::EngineEvent::Done => break,
            crate::types::EngineEvent::Error(msg) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(OpenAiErrorResponse::bad_request(msg)),
                )
                    .into_response();
            }
        }
    }

    GLOBAL_RESPONSE_CACHE.put(crate::storage::CachedResponse {
        response_id: response_id.clone(),
        output_text: out.clone(),
        output_token_ids: Vec::new(),
    });

    (
        StatusCode::OK,
        Json(ResponsesResource {
            id: response_id,
            object: "response".to_string(),
            status: "completed".to_string(),
            output_text: Some(out),
        }),
    )
        .into_response()
}

pub async fn responses_get(
    headers: HeaderMap,
    State(state): State<SharedRwkvInferState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &state.auth) {
        return resp;
    }
    let _ = state;
    match GLOBAL_RESPONSE_CACHE.get(&response_id) {
        Some(resp) => (
            StatusCode::OK,
            Json(ResponsesResource {
                id: resp.response_id,
                object: "response".to_string(),
                status: "completed".to_string(),
                output_text: Some(resp.output_text),
            }),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(OpenAiErrorResponse::bad_request("response not found")),
        )
            .into_response(),
    }
}

pub async fn responses_delete(
    headers: HeaderMap,
    State(state): State<SharedRwkvInferState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &state.auth) {
        return resp;
    }
    let _ = state;
    let removed = GLOBAL_RESPONSE_CACHE.remove(&response_id);
    if removed {
        (
            StatusCode::OK,
            Json(serde_json::json!({ "id": response_id, "deleted": true })),
        )
            .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(OpenAiErrorResponse::bad_request("response not found")),
        )
            .into_response()
    }
}

pub async fn responses_cancel(
    headers: HeaderMap,
    State(state): State<SharedRwkvInferState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &state.auth) {
        return resp;
    }
    let _ = (state, response_id);
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(OpenAiErrorResponse::not_supported(
            "cancel is not wired yet",
        )),
    )
        .into_response()
}
