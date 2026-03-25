use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::{
    access::http_api::{HttpApiService, HttpApiState},
    auth::check_api_key,
};

pub async fn images_generations(
    headers: HeaderMap,
    State(app): State<HttpApiState>,
    body: axum::body::Bytes,
) -> Response {
    let _ = body;
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
    match api.images_generations() {
        Ok(()) => Json(sonic_rs::json!({})).into_response(),
        Err(crate::Error::NotSupported(msg)) => (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(crate::access::http_api::OpenAiErrorResponse::not_supported(
                msg,
            )),
        )
            .into_response(),
        Err(e) => crate::access::http_api::infer_error_response(e),
    }
}
