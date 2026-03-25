use std::time::Duration;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::{
    access::http_api::{ChatCompletionRequest, HttpApiService, HttpApiState},
    auth::check_api_key,
};

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
    State(app): State<HttpApiState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        #[cfg(feature = "trace")]
        tracing::warn!("api key check failed");
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
    let run = match api.chat_completions(req).await {
        Ok(run) => run,
        Err(e) => return crate::access::http_api::infer_error_response(e),
    };

    if run.stream_requested {
        let (sse_tx, sse_rx) = mpsc::channel(256);
        tokio::spawn(async move {
            let mut run = run;
            #[cfg(feature = "trace")]
            let mut emitted_chunks = 0usize;
            #[cfg(feature = "trace")]
            tracing::info!(chat_completion_id = %run.id, model = %run.model, "chat stream opened");

            let role_chunk = sonic_rs::to_string(&run.stream_role_chunk()).unwrap_or_default();
            if sse_tx
                .send(axum::response::sse::Event::default().data(role_chunk))
                .await
                .is_err()
            {
                #[cfg(feature = "trace")]
                tracing::warn!(chat_completion_id = %run.id, "sse receiver dropped before role chunk");
                return;
            }

            while let Some(ev) = run.rx.recv().await {
                match ev {
                    crate::inference_core::EngineEvent::Output(delta) => {
                        #[cfg(feature = "trace")]
                        {
                            emitted_chunks += 1;
                            tracing::trace!(
                                chat_completion_id = %run.id,
                                chunk = emitted_chunks,
                                chars = delta.text.chars().count(),
                                tokens = delta.tokens.len(),
                                "chat stream chunk"
                            );
                        }
                        let chunk = run.stream_chunk(&delta, None);
                        let json = sonic_rs::to_string(&chunk).unwrap_or_default();
                        if sse_tx
                            .send(axum::response::sse::Event::default().data(json))
                            .await
                            .is_err()
                        {
                            #[cfg(feature = "trace")]
                            tracing::warn!(chat_completion_id = %run.id, "sse receiver dropped");
                            break;
                        }
                    }
                    crate::inference_core::EngineEvent::Done(meta) => {
                        #[cfg(feature = "trace")]
                        tracing::info!(
                            emitted_chunks,
                            finish_reason = meta.reason.as_openai_str(),
                            "chat completion stream done"
                        );
                        let final_chunk = run.finish_chunk(&meta);
                        let final_json = sonic_rs::to_string(&final_chunk).unwrap_or_default();
                        if sse_tx
                            .send(axum::response::sse::Event::default().data(final_json))
                            .await
                            .is_err()
                        {
                            #[cfg(feature = "trace")]
                            tracing::warn!(chat_completion_id = %run.id, "sse receiver dropped before final chunk");
                            break;
                        }
                        let _ = sse_tx
                            .send(axum::response::sse::Event::default().data("[DONE]"))
                            .await;
                        break;
                    }
                    crate::inference_core::EngineEvent::Error(msg) => {
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
            tracing::info!(chat_completion_id = %run.id, emitted_chunks, "chat stream closed");
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
        Err(e) => crate::access::http_api::infer_error_response(e),
    }
}
