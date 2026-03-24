use axum::Json;
use axum::response::IntoResponse;

use crate::dtos::admin::models::reload::ModelsReloadReq;
use crate::services::admin::models::reload::reload_models;

pub async fn admin_models_reload(Json(req): Json<ModelsReloadReq>) -> axum::response::Response {
    match reload_models(req).await {
        Ok(resp) => Json(resp).into_response(),
        Err(err) => err.into_response(),
    }
}
