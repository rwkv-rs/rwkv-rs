use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use crate::api::ApiService;
use crate::auth::check_api_key;
use crate::server::{AppState, ReloadModelsRequest};

pub async fn admin_models_reload(
    headers: HeaderMap,
    State(app): State<AppState>,
    Json(req): Json<ReloadModelsRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let api = ApiService::new(app.runtime_manager.clone());
    match api.admin_models_reload(req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => crate::server::infer_error_response(e),
    }
}
