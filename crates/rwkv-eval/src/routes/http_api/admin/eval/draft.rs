use axum::{Json, extract::State};

use crate::{
    dtos::AdminEvalConfigDto,
    routes::http_api::{AppState, error::ApiResult},
};
use super::service::build_admin_eval_draft;

#[utoipa::path(
    get,
    path = "/api/v1/admin/eval/draft",
    responses(
        (status = 200, description = "Default admin evaluation config draft", body = AdminEvalConfigDto),
        (status = 401, description = "Unauthorized", body = super::super::super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::super::super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_draft(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalConfigDto> {
    Ok(Json(build_admin_eval_draft(&state.service_config)))
}
