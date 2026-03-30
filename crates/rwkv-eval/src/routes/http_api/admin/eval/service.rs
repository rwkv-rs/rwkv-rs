use axum::Json;

use crate::{
    dtos::AdminEvalStatusResponse,
    routes::http_api::{
        AppState,
        error::{ApiError, ApiResult},
        tasks::to_task_resource,
    },
    services::{admin::EvalRunSnapshot, tasks},
};

pub(crate) async fn admin_eval_status_response(
    state: &AppState,
) -> ApiResult<AdminEvalStatusResponse> {
    let snapshot = state.eval_controller.snapshot().await?;
    Ok(Json(build_admin_eval_status(state, snapshot).await?))
}

async fn build_admin_eval_status(
    state: &AppState,
    snapshot: Option<EvalRunSnapshot>,
) -> Result<AdminEvalStatusResponse, ApiError> {
    let Some(snapshot) = snapshot else {
        return Ok(AdminEvalStatusResponse {
            status: "idle".to_string(),
            desired_state: None,
            config_path: None,
            started_at_unix_ms: None,
            updated_at_unix_ms: None,
            finished_at_unix_ms: None,
            error: None,
            tasks: Vec::new(),
            tasks_total: 0,
            attempts_planned: 0,
            attempts_completed: 0,
            progress_percent: 0.0,
        });
    };

    let rows = tasks::by_config_path(&state.db, snapshot.config_path.clone()).await?;
    let tasks = rows
        .iter()
        .map(to_task_resource)
        .collect::<Result<Vec<_>, _>>()?;
    let attempts_planned = tasks
        .iter()
        .map(|task| task.progress.planned_attempts)
        .sum::<u64>();
    let attempts_completed = tasks
        .iter()
        .map(|task| task.progress.completed_attempts)
        .sum::<u64>();
    let progress_percent = if attempts_planned == 0 {
        0.0
    } else {
        ((attempts_completed.min(attempts_planned)) as f64 / attempts_planned as f64)
            .clamp(0.0, 1.0)
    };

    Ok(AdminEvalStatusResponse {
        status: snapshot.runtime.observed_status.as_str().to_string(),
        desired_state: Some(snapshot.desired_state.as_str().to_string()),
        config_path: Some(snapshot.config_path),
        started_at_unix_ms: Some(snapshot.runtime.started_at_unix_ms),
        updated_at_unix_ms: Some(snapshot.runtime.updated_at_unix_ms),
        finished_at_unix_ms: snapshot.runtime.finished_at_unix_ms,
        error: snapshot.runtime.error,
        tasks_total: tasks.len() as u64,
        attempts_planned,
        attempts_completed,
        progress_percent,
        tasks,
    })
}
