use axum::{
    Json,
    extract::{Path, State},
};

use crate::{
    dtos::TaskDetailResponse,
    routes::http_api::{
        AppState,
        error::{ApiError, ApiResult},
    },
    services::tasks,
};
use super::to_task_resource;

#[utoipa::path(
    get,
    path = "/api/v1/tasks/{task_id}",
    params(("task_id" = i32, Path, description = "Task ID")),
    responses(
        (status = 200, description = "Task detail with derived counters", body = TaskDetailResponse),
        (status = 404, description = "Task not found", body = super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "tasks"
)]
pub(crate) async fn task_detail(
    State(state): State<AppState>,
    Path(task_id): Path<i32>,
) -> ApiResult<TaskDetailResponse> {
    let detail = tasks::detail(&state.db, task_id)
        .await?
        .ok_or_else(|| ApiError::not_found(format!("task `{task_id}` not found")))?;

    Ok(Json(TaskDetailResponse {
        task: to_task_resource(&detail.task)?,
        attempts_total: detail.attempts_total as u64,
        attempts_passed: detail.attempts_passed as u64,
        attempts_failed: detail.attempts_failed as u64,
        attempts_with_checker: detail.attempts_with_checker as u64,
        attempts_missing_checker: detail.attempts_missing_checker as u64,
        attempts_needing_human_review: detail.attempts_needing_human_review as u64,
    }))
}
