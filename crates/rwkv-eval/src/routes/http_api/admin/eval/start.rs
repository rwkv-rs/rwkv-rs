use axum::{Json, extract::State};

use crate::{
    dtos::{AdminEvalConfigDto, AdminEvalStatusResponse},
    routes::http_api::{AppState, error::ApiResult},
};
use super::service::{admin_eval_status_response, parse_admin_eval_request};

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/start",
    request_body = AdminEvalConfigDto,
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
    Json(payload): Json<AdminEvalConfigDto>,
) -> ApiResult<AdminEvalStatusResponse> {
    let run_cfg = parse_admin_eval_request(payload, &state.service_config)?;
    state.eval_controller.start(&run_cfg).await?;
    admin_eval_status_response(&state).await
}
