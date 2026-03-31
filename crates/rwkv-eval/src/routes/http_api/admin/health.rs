use axum::{Json, extract::State};

use crate::{
    dtos::AdminHealthResponse,
    routes::http_api::{AppState, error::ApiResult},
    services::admin::fetch_admin_health_targets,
};
use super::mapper::to_admin_health_target_resource;

#[utoipa::path(
    get,
    path = "/api/v1/admin/health",
    responses(
        (status = 200, description = "Current infer telemetry fan-out", body = AdminHealthResponse),
        (status = 401, description = "Unauthorized", body = super::super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_health(State(state): State<AppState>) -> ApiResult<AdminHealthResponse> {
    let snapshot = state.eval_controller.snapshot().await?;
    let targets = fetch_admin_health_targets(
        &state.health_client,
        snapshot.as_ref(),
        &state.service_config,
    )
    .await?;
    Ok(Json(AdminHealthResponse {
        targets: targets
            .into_iter()
            .map(to_admin_health_target_resource)
            .collect(),
    }))
}
