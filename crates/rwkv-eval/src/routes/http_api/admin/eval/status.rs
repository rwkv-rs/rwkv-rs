use axum::extract::State;

use crate::{
    dtos::AdminEvalStatusResponse,
    routes::http_api::{AppState, error::ApiResult},
};
use super::service::admin_eval_status_response;

#[utoipa::path(
    get,
    path = "/api/v1/admin/eval/status",
    responses(
        (status = 200, description = "Current admin evaluation process status", body = AdminEvalStatusResponse),
        (status = 401, description = "Unauthorized", body = super::super::super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_status(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    admin_eval_status_response(&state).await
}
