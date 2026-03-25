use axum::response::{IntoResponse, Response};

pub async fn responses_cancel() -> Response {
    match crate::services::responses::cancel::cancel().await {
        Ok(()) => unreachable!(),
        Err(err) => err.into_response(),
    }
}
