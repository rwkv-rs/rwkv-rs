use axum::{Json, extract::State};

use crate::{dtos::models::ModelsResp, routes::http_api::AppState};

pub async fn models(State(app_state): State<AppState>) -> Json<ModelsResp> {
    let queues = app_state
        .queues
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Json(crate::services::models::models(&queues))
}
