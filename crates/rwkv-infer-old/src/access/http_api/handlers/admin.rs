use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::{
    access::http_api::{HttpApiService, HttpApiState, ReloadModelsRequest},
    auth::check_api_key,
};

#[cfg_attr(
    feature = "trace",
    tracing::instrument(
        name = "rwkv.infer.http.admin_models_reload",
        skip_all,
        fields(path = "/admin/models/reload")
    )
)]
pub async fn admin_models_reload(
    headers: HeaderMap,
    State(app): State<HttpApiState>,
    Json(req): Json<ReloadModelsRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = HttpApiService::new(app.runtime_manager.clone());
    match api.admin_models_reload(req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => crate::access::http_api::infer_error_response(e),
    }
}
