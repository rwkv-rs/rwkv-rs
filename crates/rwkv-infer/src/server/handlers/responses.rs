use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::auth::check_api_key;
use crate::engine::SubmitOutput;
use crate::server::RwkvInferApp;
use crate::server::openai_types::OpenAiErrorResponse;
use crate::storage::{GLOBAL_BACKGROUND_TASKS, GLOBAL_RESPONSE_CACHE};
use crate::types::SamplingConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesCreateRequest {
    pub model: String,
    pub input: String,
    pub background: Option<bool>,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub top_k: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub penalty_decay: Option<f32>,
    pub stop: Option<crate::server::StopField>,
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
    State(app): State<RwkvInferApp>,
    Json(req): Json<ResponsesCreateRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth) {
        return resp;
    }
    if req.model.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(OpenAiErrorResponse::bad_request(
                "model is required and cannot be empty",
            )),
        )
            .into_response();
    }

    let response_id = GLOBAL_RESPONSE_CACHE.next_response_id();
    let background = req.background.unwrap_or(false);
    let stop_suffixes = req.stop.map(crate::server::StopField::into_vec).unwrap_or_default();

    let sampling = SamplingConfig {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        top_p: req.top_p.unwrap_or(1.0),
        max_new_tokens: req.max_output_tokens.unwrap_or(256) as usize,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        repetition_penalty: req.repetition_penalty.unwrap_or(0.0),
        penalty_decay: req.penalty_decay.unwrap_or(1.0),
    };

    if background {
        let task = GLOBAL_BACKGROUND_TASKS.create(response_id.clone());
        let service = app.service.clone();
        let model_name = req.model.clone();
        let input = req.input.clone();
        let stop_suffixes = stop_suffixes.clone();
        let response_id_for_task = response_id.clone();
        tokio::spawn(async move {
            GLOBAL_BACKGROUND_TASKS.set_state(
                &task.task_id,
                crate::storage::BackgroundTaskState::InProgress,
            );
            let submit = service
                .submit_text(&model_name, input, sampling, stop_suffixes, true)
                .await;
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
    let submit = app
        .service
        .submit_text(&req.model, req.input, sampling, stop_suffixes, true)
        .await;
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
            let status = match e {
                crate::Error::BadRequest(_) | crate::Error::NotSupported(_) => StatusCode::BAD_REQUEST,
                crate::Error::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return (
                status,
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
    State(app): State<RwkvInferApp>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth) {
        return resp;
    }
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
    State(app): State<RwkvInferApp>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth) {
        return resp;
    }
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
    State(app): State<RwkvInferApp>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth) {
        return resp;
    }
    let _ = (app, response_id);
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(OpenAiErrorResponse::not_supported(
            "cancel is not wired yet",
        )),
    )
        .into_response()
}
