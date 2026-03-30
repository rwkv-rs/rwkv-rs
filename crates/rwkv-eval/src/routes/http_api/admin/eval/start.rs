use axum::extract::State;

use crate::{
    dtos::AdminEvalStatusResponse,
    routes::http_api::{AppState, error::ApiResult},
};
use super::service::admin_eval_status_response;

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/start",
    responses(
        (status = 200, description = "Started or reused the current admin evaluation process", body = AdminEvalStatusResponse),
        (status = 400, description = "Bad request", body = super::super::super::error::ErrorResponse),
        (status = 401, description = "Unauthorized", body = super::super::super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::super::super::error::ErrorResponse),
        (status = 409, description = "Conflict", body = super::super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_start(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    state.eval_controller.start(&state.service_config).await?;
    admin_eval_status_response(&state).await
}
