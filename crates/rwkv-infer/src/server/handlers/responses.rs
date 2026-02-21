use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::api::ApiService;
use crate::auth::check_api_key;
use crate::server::{AppState, OpenAiErrorResponse, ResponseIdRequest, ResponsesCreateRequest};

#[cfg_attr(
    feature = "trace",
    tracing::instrument(
        name = "rwkv.infer.http.responses_create",
        skip_all,
        fields(path = "/v1/responses")
    )
)]
pub async fn responses_create(
    headers: HeaderMap,
    State(app): State<AppState>,
    Json(req): Json<ResponsesCreateRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    match api.responses_create(req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => crate::server::infer_error_response(e),
    }
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(
        name = "rwkv.infer.http.responses_get",
        skip_all,
        fields(path = "/v1/responses/:id")
    )
)]
pub async fn responses_get(
    headers: HeaderMap,
    State(app): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    match api.responses_get(ResponseIdRequest { response_id }) {
        Some(resp) => Json(resp).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(OpenAiErrorResponse::bad_request("response not found")),
        )
            .into_response(),
    }
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(
        name = "rwkv.infer.http.responses_delete",
        skip_all,
        fields(path = "/v1/responses/:id")
    )
)]
pub async fn responses_delete(
    headers: HeaderMap,
    State(app): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    match api.responses_delete(ResponseIdRequest {
        response_id: response_id.clone(),
    }) {
        Some(resp) => Json(resp).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(OpenAiErrorResponse::bad_request("response not found")),
        )
            .into_response(),
    }
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(
        name = "rwkv.infer.http.responses_cancel",
        skip_all,
        fields(path = "/v1/responses/:id/cancel")
    )
)]
pub async fn responses_cancel(
    headers: HeaderMap,
    State(app): State<AppState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    match api.responses_cancel(ResponseIdRequest { response_id }) {
        Ok(()) => Json(sonic_rs::json!({ "ok": true })).into_response(),
        Err(crate::Error::NotSupported(msg)) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(OpenAiErrorResponse::not_supported(msg)),
        )
            .into_response(),
        Err(e) => crate::server::infer_error_response(e),
    }
}
