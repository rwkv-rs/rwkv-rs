use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::auth::check_api_key;
use crate::server::AppState;
use crate::server::openai_types::{OpenAiErrorResponse, ReloadModelsRequest, ReloadModelsResponse};
use crate::service::runtime_manager::ModelsReloadPatch;

pub async fn admin_models_reload(
    headers: HeaderMap,
    State(app): State<AppState>,
    Json(req): Json<ReloadModelsRequest>,
) -> Response {
    if let Err(resp) = check_api_key(&headers, &app.auth_cfg) {
        return resp;
    }

    let patch = ModelsReloadPatch {
        upsert: req.upsert,
        remove_model_names: req.remove_model_names,
        dry_run: req.dry_run.unwrap_or(false),
    };

    match app.runtime_manager.reload_models(patch).await {
        Ok(result) => (
            StatusCode::OK,
            Json(ReloadModelsResponse {
                changed_model_names: result.changed_model_names,
                rebuilt_model_names: result.rebuilt_model_names,
                removed_model_names: result.removed_model_names,
                active_model_names: result.active_model_names,
                dry_run: result.dry_run,
                message: result.message,
            }),
        )
            .into_response(),
        Err(e) => {
            let status = match e {
                crate::Error::BadRequest(_) | crate::Error::NotSupported(_) => {
                    StatusCode::BAD_REQUEST
                }
                crate::Error::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(OpenAiErrorResponse::bad_request(e.to_string())),
            )
                .into_response()
        }
    }
}
