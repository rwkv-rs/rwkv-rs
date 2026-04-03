use axum::{extract::State, response::IntoResponse};

use crate::{
    dtos::admin::models::reload::ModelsReloadReq,
    routes::http_api::AppState,
    services::admin::models::reload::reload_models,
    sonic_json::SonicJson,
};

pub async fn admin_models_reload(
    State(app_state): State<AppState>,
    SonicJson(req): SonicJson<ModelsReloadReq>,
) -> axum::response::Response {
    match reload_models(
        &app_state.queues,
        app_state.reload_lock.as_ref(),
        app_state.infer_cfg_path.as_deref(),
        app_state.build_queues.as_ref(),
        req,
    )
    .await
    {
        Ok(resp) => SonicJson(resp).into_response(),
        Err(err) => err.into_response(),
    }
}
