use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::cores::queue::QueueEvent;
use crate::dtos::chat::completions::ChatCompletionsReq;
use crate::routes::AppState;
use crate::services::chat::completions::chat_completions as run_chat_completions;

pub async fn chat_completions(
    State(app_state): State<AppState>,
    Json(req): Json<ChatCompletionsReq>,
) -> Response {
    let run = match run_chat_completions(app_state.clone(), req).await {
        Ok(run) => run,
        Err(err) => return err.into_response(),
    };

    if run.stream_requested {
        let keep_alive = KeepAlive::new().interval(Duration::from_millis(app_state.sse_keep_alive_ms));
        let (sse_tx, sse_rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let mut run = run;
            let mut stream_state = run.new_stream_state();
            let role_chunk = sonic_rs::to_string(&run.stream_role_chunk()).unwrap();
            if sse_tx.send(Event::default().data(role_chunk)).await.is_err() {
                return;
            }

            let mut finish_meta = None;
            while let Some(event) = run.rx.recv().await {
                match event {
                    QueueEvent::Delta(delta) => {
                        for chunk in run.stream_chunks(&mut stream_state, &delta) {
                            let json = sonic_rs::to_string(&chunk).unwrap();
                            if sse_tx.send(Event::default().data(json)).await.is_err() {
                                return;
                            }
                        }
                    }
                    QueueEvent::Done(meta) => {
                        finish_meta = Some(meta);
                        break;
                    }
                }
            }

            let finish_meta = finish_meta.expect("queue stream closed without finish meta");
            for chunk in run.finish_chunks(&mut stream_state, &finish_meta) {
                let json = sonic_rs::to_string(&chunk).unwrap();
                let _ = sse_tx.send(Event::default().data(json)).await;
            }
            let _ = sse_tx.send(Event::default().data("[DONE]")).await;
        });

        let sse_stream = ReceiverStream::new(sse_rx).map(Ok::<_, std::convert::Infallible>);
        return Sse::new(sse_stream).keep_alive(keep_alive).into_response();
    }

    Json(run.collect().await).into_response()
}
