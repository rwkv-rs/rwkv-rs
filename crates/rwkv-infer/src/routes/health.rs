use axum::Json;

use crate::dtos::health::HealthResp;

pub async fn health() -> Json<HealthResp> {
    Json(crate::services::health::health())
}
