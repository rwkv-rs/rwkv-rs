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
use crate::server::{AppState, CompletionRequest};

pub async fn completions(
    headers: HeaderMap,
    State(app): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    let run = match api.completions(req).await {
        Ok(run) => run,
        Err(e) => return crate::server::infer_error_response(e),
    };

    if run.stream_requested {
        let crate::api::CompletionRun {
            id,
            created,
            model,
            stream_requested: _,
            rx,
        } = run;

        let (sse_tx, sse_rx) = mpsc::channel(256);
        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(ev) = rx.recv().await {
                match ev {
                    crate::types::EngineEvent::Text(text) => {
                        let chunk = crate::server::CompletionResponse {
                            id: id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::CompletionResponseChoice {
                                text,
                                index: 0,
                                finish_reason: None,
                            }],
                        };
                        let json = sonic_rs::to_string(&chunk).unwrap_or_default();
                        if sse_tx
                            .send(axum::response::sse::Event::default().data(json))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    crate::types::EngineEvent::Done(meta) => {
                        let final_chunk = crate::server::CompletionResponse {
                            id: id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::CompletionResponseChoice {
                                text: String::new(),
                                index: 0,
                                finish_reason: Some(meta.reason.as_openai_str().to_string()),
                            }],
                        };
                        let final_json = sonic_rs::to_string(&final_chunk).unwrap_or_default();
                        if sse_tx
                            .send(axum::response::sse::Event::default().data(final_json))
                            .await
                            .is_err()
                        {
                            break;
                        }
                        let _ = sse_tx
                            .send(axum::response::sse::Event::default().data("[DONE]"))
                            .await;
                        break;
                    }
                    crate::types::EngineEvent::Error(msg) => {
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
