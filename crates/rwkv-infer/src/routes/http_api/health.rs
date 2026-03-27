use axum::{Json, extract::State};

use crate::{dtos::health::HealthResp, routes::http_api::AppState};

pub async fn health(State(app_state): State<AppState>) -> Json<HealthResp> {
    let queues = app_state
        .queues
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Json(crate::services::health::health(
        &queues,
        app_state.gpu_metrics.as_ref(),
    ))
}
