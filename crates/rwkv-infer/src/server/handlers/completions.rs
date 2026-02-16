use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::auth::check_api_key;
use crate::engine::SubmitOutput;
use crate::server::RwkvInferApp;
use crate::server::openai_types::{
    CompletionRequest, CompletionResponse, CompletionResponseChoice, OpenAiErrorResponse, StopField,
};
use crate::types::SamplingConfig;

pub async fn completions(
    headers: HeaderMap,
    State(app): State<RwkvInferApp>,
    Json(req): Json<CompletionRequest>,
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

    let stream = req.stream.unwrap_or(false);
    let sampling = SamplingConfig {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        top_p: req.top_p.unwrap_or(1.0),
        max_new_tokens: req.max_tokens.unwrap_or(256) as usize,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        repetition_penalty: req.repetition_penalty.unwrap_or(0.0),
        penalty_decay: req.penalty_decay.unwrap_or(1.0),
    };
    let stop_suffixes = req.stop.map(StopField::into_vec).unwrap_or_default();

    // Always request an event stream from the engine; non-streaming responses just collect it.
    let submit = app
        .service
        .submit_text(&req.model, req.prompt, sampling, stop_suffixes, true)
        .await;

    let (entry_id, rx) = match submit {
        Ok(SubmitOutput::Stream { entry_id, rx }) => (entry_id, rx),
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

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model = req.model;
    let id = format!("cmpl_{}", Uuid::new_v4().as_simple());

    if stream {
        let sse_stream = ReceiverStream::new(rx).map(move |ev| match ev {
            crate::types::EngineEvent::Text(text) => {
                let chunk = CompletionResponse {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![CompletionResponseChoice {
                        text,
                        index: 0,
                        finish_reason: None,
                    }],
                };
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(json))
            }
            crate::types::EngineEvent::Done => Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default().data("[DONE]"),
            ),
            crate::types::EngineEvent::Error(msg) => Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default().data(format!(
                    "{{\"error\":{}}}",
                    serde_json::to_string(&msg).unwrap_or_default()
                )),
            ),
        });

        let keep_alive = axum::response::sse::KeepAlive::new().interval(
            std::time::Duration::from_millis(app.cfg.sse_keep_alive_ms),
        );
        return axum::response::Sse::new(sse_stream)
            .keep_alive(keep_alive)
            .into_response();
    }

    // Non-streaming: collect until done.
    let mut text_out = String::new();
    let mut rx = rx;
    while let Some(ev) = rx.recv().await {
        match ev {
            crate::types::EngineEvent::Text(t) => text_out.push_str(&t),
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

    let resp = CompletionResponse {
        id,
        object: "text_completion".to_string(),
        created,
        model,
        choices: vec![CompletionResponseChoice {
            text: text_out,
            index: 0,
            finish_reason: Some("stop".to_string()),
        }],
    };

    let _ = entry_id; // reserved for future responses API integration.
    (StatusCode::OK, Json(resp)).into_response()
}
