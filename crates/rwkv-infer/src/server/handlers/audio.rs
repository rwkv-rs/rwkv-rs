use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::auth::check_api_key;
use crate::server::AppState;
use crate::server::openai_types::OpenAiErrorResponse;

pub async fn audio_speech(
    headers: HeaderMap,
    State(app): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let _ = body;
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(OpenAiErrorResponse::not_supported(
            "/v1/audio/speech not implemented yet",
        )),
    )
        .into_response()
}
