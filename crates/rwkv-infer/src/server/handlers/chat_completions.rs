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
use crate::config::SamplingConfig;
use crate::engine::SubmitOutput;
use crate::server::SharedRwkvInferState;
use crate::server::openai_types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
    OpenAiErrorResponse,
};

pub async fn chat_completions(
    headers: HeaderMap,
    State(state): State<SharedRwkvInferState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &state.auth) {
        return resp;
    }

    let stream = req.stream.unwrap_or(false);
    let sampling = SamplingConfig {
        temperature: req.temperature.unwrap_or(1.0),
        top_k: 0,
        top_p: req.top_p.unwrap_or(1.0),
        max_new_tokens: req.max_tokens.unwrap_or(256) as usize,
    };

    // Minimal chat formatting: concatenate messages as "role: content\n".
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&msg.role);
        prompt.push_str(": ");
        prompt.push_str(&msg.content);
        prompt.push('\n');
    }

    // Always request an event stream from the engine; non-streaming responses just collect it.
    let submit = state.engine.submit_text(prompt, sampling, true).await;
    let (_entry_id, rx) = match submit {
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
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::bad_request(e.to_string())),
            )
                .into_response();
        }
    };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model = req.model.unwrap_or_else(|| "rwkv".to_string());
    let id = format!("chatcmpl_{}", Uuid::new_v4().as_simple());

    if stream {
        let sse_stream = ReceiverStream::new(rx).map(move |ev| match ev {
            crate::types::EngineEvent::Text(text) => {
                let chunk = ChatCompletionResponse {
                    id: id.clone(),
                    object: "chat.completion".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionResponseChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text,
                        },
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
            std::time::Duration::from_millis(state.cfg.sse_keep_alive_ms),
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

    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![ChatCompletionResponseChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text_out,
            },
            finish_reason: Some("stop".to_string()),
        }],
    };

    (StatusCode::OK, Json(resp)).into_response()
}
