use std::time::Duration;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::ApiService;
use crate::auth::check_api_key;
use crate::server::{AppState, ChatCompletionRequest};

#[cfg_attr(
    feature = "trace",
    tracing::instrument(
        name = "rwkv.infer.http.chat_completions",
        skip_all,
        fields(path = "/v1/chat/completions")
    )
)]
pub async fn chat_completions(
    headers: HeaderMap,
    State(app): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        #[cfg(feature = "trace")]
        tracing::warn!("api key check failed");
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    let run = match api.chat_completions(req).await {
        Ok(run) => run,
        Err(e) => return crate::server::infer_error_response(e),
    };

    if run.stream_requested {
        let crate::api::ChatCompletionRun {
            id,
            created,
            model,
            stream_requested: _,
            rx,
        } = run;

        let (sse_tx, sse_rx) = mpsc::channel(256);
        tokio::spawn(async move {
            let mut rx = rx;
            #[cfg(feature = "trace")]
            let mut emitted_chunks = 0usize;
            #[cfg(feature = "trace")]
            tracing::info!(chat_completion_id = %id, model = %model, "chat stream opened");
            while let Some(ev) = rx.recv().await {
                match ev {
                    crate::types::EngineEvent::Text(text) => {
                        #[cfg(feature = "trace")]
                        {
                            emitted_chunks += 1;
                            tracing::trace!(
                                chat_completion_id = %id,
                                chunk = emitted_chunks,
                                chars = text.chars().count(),
                                "chat stream chunk"
                            );
                        }
                        let chunk = crate::server::ChatCompletionResponse {
                            id: id.clone(),
                            object: "chat.completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::ChatCompletionResponseChoice {
                                index: 0,
                                message: crate::server::ChatMessage {
                                    role: "assistant".to_string(),
                                    content: text,
                                },
                                finish_reason: None,
                            }],
                            timings_ms: None,
                        };
                        let json = sonic_rs::to_string(&chunk).unwrap_or_default();
                        if sse_tx
                            .send(axum::response::sse::Event::default().data(json))
                            .await
                            .is_err()
                        {
                            #[cfg(feature = "trace")]
                            tracing::warn!(chat_completion_id = %id, "sse receiver dropped");
                            break;
                        }
                    }
                    crate::types::EngineEvent::Done(meta) => {
                        #[cfg(feature = "trace")]
                        tracing::info!(
                            emitted_chunks,
                            finish_reason = meta.reason.as_openai_str(),
                            "chat completion stream done"
                        );
                        let final_chunk = crate::server::ChatCompletionResponse {
                            id: id.clone(),
                            object: "chat.completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::ChatCompletionResponseChoice {
                                index: 0,
                                message: crate::server::ChatMessage {
                                    role: "assistant".to_string(),
                                    content: String::new(),
                                },
                                finish_reason: Some(meta.reason.as_openai_str().to_string()),
                            }],
                            timings_ms: meta.timings_ms.clone(),
                        };
                        let final_json = sonic_rs::to_string(&final_chunk).unwrap_or_default();
                        if sse_tx
                            .send(axum::response::sse::Event::default().data(final_json))
                            .await
                            .is_err()
                        {
                            #[cfg(feature = "trace")]
                            tracing::warn!(chat_completion_id = %id, "sse receiver dropped before final chunk");
                            break;
                        }
                        let _ = sse_tx
                            .send(axum::response::sse::Event::default().data("[DONE]"))
                            .await;
                        break;
                    }
                    crate::types::EngineEvent::Error(msg) => {
                        #[cfg(feature = "trace")]
                        tracing::error!(error = %msg, "chat completion stream error");
                        let _ = sse_tx
                            .send(axum::response::sse::Event::default().data(format!(
                                "{{\"error\":{}}}",
                                sonic_rs::to_string(&msg).unwrap_or_default()
                            )))
                            .await;
                        break;
                    }
                }
            }
            #[cfg(feature = "trace")]
            tracing::info!(chat_completion_id = %id, emitted_chunks, "chat stream closed");
        });
        let sse_stream = ReceiverStream::new(sse_rx).map(Ok::<_, std::convert::Infallible>);

        let keep_alive = axum::response::sse::KeepAlive::new().interval(Duration::from_millis(
            app.runtime_manager.sse_keep_alive_ms(),
        ));
        return axum::response::Sse::new(sse_stream)
            .keep_alive(keep_alive)
            .into_response();
    }

    match run.collect().await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => crate::server::infer_error_response(e),
    }
}
