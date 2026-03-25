pub mod handlers;
mod openai_schema;
mod router;
mod service;
pub mod validation;

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
pub use openai_schema::*;
pub use router::{AppState, HttpApiRouterBuilder, HttpApiState, RouterBuilder};
pub use service::{ApiService, HttpApiService, collect_stream_output};

pub fn infer_error_response(err: crate::Error) -> Response {
    let status = if err.is_client_error() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    let chain = err.format_chain();
    if err.is_client_error() {
        log::warn!("request failed: {chain}");
        #[cfg(feature = "trace")]
        tracing::warn!(error = %chain, "request failed");
    } else {
        log::error!("request failed: {chain}");
        #[cfg(feature = "trace")]
        tracing::error!(error = %chain, "request failed");
    }
    (status, Json(OpenAiErrorResponse::from_infer_error(&err))).into_response()
}
