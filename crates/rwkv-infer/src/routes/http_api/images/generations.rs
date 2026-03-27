use axum::response::{IntoResponse, Response};

pub async fn images_generations() -> Response {
    match crate::services::images::generations::generations().await {
        Ok(()) => unreachable!(),
        Err(err) => err.into_response(),
    }
}
