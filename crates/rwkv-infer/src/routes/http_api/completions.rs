use std::time::Duration;

use axum::{
    extract::State,
    response::{
        IntoResponse,
        Response,
        Sse,
        sse::{Event, KeepAlive},
    },
};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::{
    cores::queue::QueueEvent,
    dtos::completions::CompletionsReq,
    routes::http_api::AppState,
    services::{completions::completions as run_completions, select_model_queue},
    sonic_json::SonicJson,
};

pub async fn completions(
    State(app_state): State<AppState>,
    SonicJson(req): SonicJson<CompletionsReq>,
) -> Response {
    let handle = {
        let queues = app_state
            .queues
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match select_model_queue(&queues, &req.model) {
            Ok(handle) => handle,
            Err(err) => return err.into_response(),
        }
    };
    let run = match run_completions(handle, req).await {
        Ok(run) => run,
        Err(err) => return err.into_response(),
    };

    if run.stream_requested {
        let keep_alive =
            KeepAlive::new().interval(Duration::from_millis(app_state.sse_keep_alive_ms));
        let (sse_tx, sse_rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let mut run = run;
            let mut text_offset = 0usize;
            let mut finish_meta = None;
            while let Some(event) = run.rx.recv().await {
                match event {
                    QueueEvent::Delta(delta) => {
                        let json =
                            sonic_rs::to_string(&run.stream_chunk(&delta, text_offset)).unwrap();
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

    SonicJson(run.collect().await).into_response()
}
