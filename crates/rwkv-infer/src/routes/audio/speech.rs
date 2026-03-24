use axum::response::{IntoResponse, Response};

pub async fn audio_speech() -> Response {
    match crate::services::audio::speech::speech().await {
        Ok(()) => unreachable!(),
        Err(err) => err.into_response(),
    }
}
