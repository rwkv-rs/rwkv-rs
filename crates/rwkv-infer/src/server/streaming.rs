use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::types::EngineEvent;

pub fn sse_from_engine_events(
    rx: tokio::sync::mpsc::Receiver<EngineEvent>,
    keep_alive_ms: u64,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let stream = ReceiverStream::new(rx).map(|ev| match ev {
        EngineEvent::Text(text) => Ok(Event::default().data(text)),
        EngineEvent::Done => Ok(Event::default().data("[DONE]")),
        EngineEvent::Error(msg) => Ok(Event::default().data(format!("error: {msg}"))),
    });

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_ms)))
}
