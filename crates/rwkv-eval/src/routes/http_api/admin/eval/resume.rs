use axum::extract::State;

use crate::{
    dtos::AdminEvalStatusResponse,
    routes::http_api::{AppState, error::ApiResult},
};
use super::service::admin_eval_status_response;

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/resume",
    responses(
        (status = 200, description = "Resumed a paused evaluation", body = AdminEvalStatusResponse),
        (status = 401, description = "Unauthorized", body = super::super::super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::super::super::error::ErrorResponse),
        (status = 404, description = "No active evaluation", body = super::super::super::error::ErrorResponse),
        (status = 409, description = "Conflict", body = super::super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_resume(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    state.eval_controller.resume().await?;
    admin_eval_status_response(&state).await
}
