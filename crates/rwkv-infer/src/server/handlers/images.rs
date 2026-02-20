use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::api::ApiService;
use crate::auth::check_api_key;
use crate::server::AppState;

pub async fn images_generations(
    headers: HeaderMap,
    State(app): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let _ = body;
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    match api.images_generations() {
        Ok(()) => Json(sonic_rs::json!({})).into_response(),
        Err(crate::Error::NotSupported(msg)) => (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(crate::server::OpenAiErrorResponse::not_supported(msg)),
        )
            .into_response(),
        Err(e) => crate::server::infer_error_response(e),
    }
}
