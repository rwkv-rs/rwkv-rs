use axum::response::{IntoResponse, Response};

pub async fn embeddings() -> Response {
    match crate::services::embeddings::embeddings().await {
        Ok(()) => unreachable!(),
        Err(err) => err.into_response(),
    }
}
