use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::auth::check_api_key;
use crate::server::SharedRwkvInferState;
use crate::server::openai_types::OpenAiErrorResponse;

pub async fn embeddings(
    headers: HeaderMap,
    State(state): State<SharedRwkvInferState>,
    body: axum::body::Bytes,
) -> Response {
    let _ = body;
    if let Err(resp) = check_api_key(&headers, &state.auth) {
        return resp;
    }
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(OpenAiErrorResponse::not_supported(
            "/v1/embeddings not implemented yet",
        )),
    )
        .into_response()
}
