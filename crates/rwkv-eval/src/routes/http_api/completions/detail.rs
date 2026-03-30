use axum::{
    Json,
    extract::{Path, State},
};

use crate::{
    dtos::CompletionDetailResponse,
    routes::http_api::{
        AppState,
        error::{ApiError, ApiResult},
    },
    services::completions,
};
use super::mapper::to_completion_detail_response;

#[utoipa::path(
    get,
    path = "/api/v1/completions/{completions_id}",
    params(("completions_id" = i32, Path, description = "Completion row ID")),
    responses(
        (status = 200, description = "Full completion/eval/checker detail", body = CompletionDetailResponse),
        (status = 404, description = "Completion not found", body = super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "attempts"
)]
pub(crate) async fn completion_detail(
    State(state): State<AppState>,
    Path(completions_id): Path<i32>,
) -> ApiResult<CompletionDetailResponse> {
    let detail = completions::detail(&state.db, completions_id)
        .await?
        .ok_or_else(|| ApiError::not_found(format!("completion `{completions_id}` not found")))?;

    Ok(Json(to_completion_detail_response(detail)?))
}
