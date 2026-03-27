use axum::response::{IntoResponse, Response};

pub mod cancel;

pub async fn responses_create() -> Response {
    crate::services::ServiceError::not_supported(
        "responses are not supported in the new HTTP path yet",
    )
    .into_response()
}

pub async fn responses_get() -> Response {
    crate::services::ServiceError::not_supported(
        "responses are not supported in the new HTTP path yet",
    )
    .into_response()
}

pub async fn responses_delete() -> Response {
    crate::services::ServiceError::not_supported(
        "responses are not supported in the new HTTP path yet",
    )
    .into_response()
}
