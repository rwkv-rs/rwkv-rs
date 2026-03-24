use axum::Json;
use axum::extract::State;

use crate::dtos::models::ModelsResp;
use crate::routes::AppState;

pub async fn models(State(app_state): State<AppState>) -> Json<ModelsResp> {
    Json(crate::services::models::models(app_state).await)
}
