use axum::{
    Json,
    extract::{Path, Query, State},
};

use crate::{
    dtos::{TaskAttemptsParams, TaskAttemptsResponse},
    routes::http_api::{AppState, error::ApiResult},
    services::tasks,
};
use super::to_task_attempt_resource;

#[utoipa::path(
    get,
    path = "/api/v1/tasks/{task_id}/attempts",
    params(
        ("task_id" = i32, Path, description = "Task ID"),
        TaskAttemptsParams
    ),
    responses(
        (status = 200, description = "Persisted attempts for a task", body = TaskAttemptsResponse),
        (status = 400, description = "Bad request", body = super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "tasks"
)]
pub(crate) async fn task_attempts(
    State(state): State<AppState>,
    Path(task_id): Path<i32>,
    Query(params): Query<TaskAttemptsParams>,
) -> ApiResult<TaskAttemptsResponse> {
    let listing = tasks::attempts(&state.db, task_id, params).await?;
    let items = listing
        .rows
        .iter()
        .map(to_task_attempt_resource)
        .collect::<Vec<_>>();

    Ok(Json(TaskAttemptsResponse {
        items,
        limit: listing.limit,
        offset: listing.offset,
        has_more: listing.has_more,
    }))
}
