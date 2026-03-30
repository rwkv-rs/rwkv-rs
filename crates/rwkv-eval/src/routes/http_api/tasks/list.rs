use axum::{
    Json,
    extract::{Query, State},
};

use crate::{
    dtos::{ListTasksParams, TaskListResponse},
    routes::http_api::{AppState, error::ApiResult},
    services::tasks,
};
use super::to_task_resource;

#[utoipa::path(
    get,
    path = "/api/v1/tasks",
    params(ListTasksParams),
    responses(
        (status = 200, description = "Joined task list for dashboard and history views", body = TaskListResponse),
        (status = 400, description = "Bad request", body = super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "tasks"
)]
pub(crate) async fn tasks(
    State(state): State<AppState>,
    Query(params): Query<ListTasksParams>,
) -> ApiResult<TaskListResponse> {
    let listing = tasks::list(&state.db, params).await?;
    let mapped = listing
        .rows
        .iter()
        .map(to_task_resource)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Json(TaskListResponse {
        items: mapped,
        limit: listing.limit,
        offset: listing.offset,
        has_more: listing.has_more,
    }))
}
