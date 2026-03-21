use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::access::http_api::HttpApiService;
use crate::access::http_api::{
    HttpApiState, OpenAiErrorResponse, ResponseIdRequest, ResponsesCreateRequest,
};
use crate::auth::check_api_key;

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
    State(app): State<HttpApiState>,
    Json(req): Json<ResponsesCreateRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
    match api.responses_create(req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => crate::access::http_api::infer_error_response(e),
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
    State(app): State<HttpApiState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
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
    State(app): State<HttpApiState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
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
    State(app): State<HttpApiState>,
    Path(response_id): Path<String>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
    match api.responses_cancel(ResponseIdRequest { response_id }) {
        Ok(()) => Json(sonic_rs::json!({ "ok": true })).into_response(),
        Err(crate::Error::NotSupported(msg)) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(OpenAiErrorResponse::not_supported(msg)),
        )
            .into_response(),
        Err(e) => crate::access::http_api::infer_error_response(e),
    }
}
