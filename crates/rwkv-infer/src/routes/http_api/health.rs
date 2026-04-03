use axum::extract::State;

use crate::{dtos::health::HealthResp, routes::http_api::AppState, sonic_json::SonicJson};

pub async fn health(State(app_state): State<AppState>) -> SonicJson<HealthResp> {
    let queues = app_state
        .queues
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    SonicJson(crate::services::health::health(
        &queues,
        app_state.gpu_metrics.as_ref(),
    ))
}
