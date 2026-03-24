use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::cores::queue::QueueEvent;
use crate::dtos::completions::CompletionsReq;
use crate::routes::AppState;
use crate::services::completions::completions as run_completions;

pub async fn completions(
    State(app_state): State<AppState>,
    Json(req): Json<CompletionsReq>,
) -> Response {
    let run = match run_completions(app_state.clone(), req).await {
        Ok(run) => run,
        Err(err) => return err.into_response(),
    };

    if run.stream_requested {
        let keep_alive = KeepAlive::new().interval(Duration::from_millis(app_state.sse_keep_alive_ms));
        let (sse_tx, sse_rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let mut run = run;
            let mut text_offset = 0usize;
            let mut finish_meta = None;
            while let Some(event) = run.rx.recv().await {
                match event {
                    QueueEvent::Delta(delta) => {
                        let json = sonic_rs::to_string(&run.stream_chunk(&delta, text_offset)).unwrap();
                        if sse_tx.send(Event::default().data(json)).await.is_err() {
                            return;
                        }
                        text_offset += delta.text.chars().count();
                    }
                    QueueEvent::Done(meta) => {
                        finish_meta = Some(meta);
                        break;
                    }
                }
            }

            let finish_meta = finish_meta.expect("queue stream closed without finish meta");
            let final_json = sonic_rs::to_string(&run.finish_chunk(&finish_meta)).unwrap();
            let _ = sse_tx.send(Event::default().data(final_json)).await;
            let _ = sse_tx.send(Event::default().data("[DONE]")).await;
        });

        let sse_stream = ReceiverStream::new(sse_rx).map(Ok::<_, std::convert::Infallible>);
        return Sse::new(sse_stream).keep_alive(keep_alive).into_response();
    }

    Json(run.collect().await).into_response()
}
