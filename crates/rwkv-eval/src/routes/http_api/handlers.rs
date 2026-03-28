use axum::{
    Json,
    extract::{Path, Query, State},
};
use utoipa::OpenApi;

use super::{
    catalog::benchmark_names_for_field,
    error::{ApiError, ApiResult},
    mappers::{
        to_admin_health_target_resource,
        to_benchmark_resource,
        to_completion_detail_response,
        to_model_resource,
        to_review_queue_resource,
        to_task_attempt_resource,
        to_task_resource,
    },
    openapi::ApiDoc,
    state::AppState,
};
use crate::{
    db::{
        ReviewQueueQuery,
        TaskAttemptsQuery,
        TaskListQuery,
        get_completion_detail,
        get_task_detail,
        list_benchmarks,
        list_models,
        list_review_queue,
        list_task_attempts,
        list_tasks,
    },
    dtos::{
        AdminEvalStatusResponse,
        AdminHealthResponse,
        ApiCompletionStatus,
        ApiCotMode,
        ApiTaskStatus,
        BenchmarkField,
        BenchmarkResource,
        CompletionDetailResponse,
        HealthResponse,
        IndexResponse,
        ListTasksParams,
        MetaResponse,
        ModelResource,
        ReviewQueueParams,
        ReviewQueueResponse,
        TaskAttemptsParams,
        TaskAttemptsResponse,
        TaskDetailResponse,
        TaskListResponse,
    },
    services::admin::{eval::EvalRunSnapshot, health::fetch_health_targets},
};

const DEFAULT_LIMIT: i64 = 50;
const MAX_LIMIT: i64 = 200;

#[utoipa::path(
    get,
    path = "/",
    responses((status = 200, description = "API index", body = IndexResponse)),
    tag = "system"
)]
pub(crate) async fn index() -> Json<IndexResponse> {
    Json(IndexResponse {
        service: "rwkv-lm-eval-api",
        docs_url: "/openapi.json",
        openapi_url: "/openapi.json",
    })
}

#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "Health check", body = HealthResponse)),
    tag = "system"
)]
pub(crate) async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[utoipa::path(
    get,
    path = "/openapi.json",
    responses((status = 200, description = "OpenAPI specification")),
    tag = "system"
)]
pub(crate) async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

#[utoipa::path(
    get,
    path = "/api/v1/meta",
    responses(
        (status = 200, description = "Dashboard metadata", body = MetaResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "meta"
)]
pub(crate) async fn meta(State(state): State<AppState>) -> ApiResult<MetaResponse> {
    let model_rows = list_models(&state.db).await.map_err(ApiError::internal)?;
    let benchmark_rows = list_benchmarks(&state.db)
        .await
        .map_err(ApiError::internal)?;

    let models = model_rows.iter().map(to_model_resource).collect();
    let benchmarks = benchmark_rows.iter().map(to_benchmark_resource).collect();

    Ok(Json(MetaResponse {
        fields: vec![
            BenchmarkField::Knowledge,
            BenchmarkField::Maths,
            BenchmarkField::Coding,
            BenchmarkField::InstructionFollowing,
            BenchmarkField::FunctionCalling,
            BenchmarkField::Unknown,
        ],
        task_statuses: vec![
            ApiTaskStatus::Running,
            ApiTaskStatus::Completed,
            ApiTaskStatus::Failed,
        ],
        completion_statuses: vec![
            ApiCompletionStatus::Running,
            ApiCompletionStatus::Completed,
            ApiCompletionStatus::Failed,
        ],
        cot_modes: vec![ApiCotMode::NoCot, ApiCotMode::FakeCot, ApiCotMode::Cot],
        models,
        benchmarks,
    }))
}

#[utoipa::path(
    get,
    path = "/api/v1/models",
    responses(
        (status = 200, description = "Registered models", body = [ModelResource]),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "meta"
)]
pub(crate) async fn models(State(state): State<AppState>) -> ApiResult<Vec<ModelResource>> {
    let rows = list_models(&state.db).await.map_err(ApiError::internal)?;
    Ok(Json(rows.iter().map(to_model_resource).collect()))
}

#[utoipa::path(
    get,
    path = "/api/v1/benchmarks",
    responses(
        (status = 200, description = "Registered benchmarks", body = [BenchmarkResource]),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "meta"
)]
pub(crate) async fn benchmarks(State(state): State<AppState>) -> ApiResult<Vec<BenchmarkResource>> {
    let rows = list_benchmarks(&state.db)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(rows.iter().map(to_benchmark_resource).collect()))
}

#[utoipa::path(
    get,
    path = "/api/v1/tasks",
    params(ListTasksParams),
    responses(
        (status = 200, description = "Joined task list for dashboard and history views", body = TaskListResponse),
        (status = 400, description = "Bad request", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "tasks"
)]
pub(crate) async fn tasks(
    State(state): State<AppState>,
    Query(params): Query<ListTasksParams>,
) -> ApiResult<TaskListResponse> {
    let (limit, fetch_limit) = resolve_limit(params.limit)?;
    let offset = resolve_offset(params.offset);
    let query = TaskListQuery {
        limit: fetch_limit,
        offset: i64::from(offset),
        config_path: None,
        latest_only: params.latest_only.unwrap_or(false),
        status: params.status.map(ApiTaskStatus::into_db),
        cot_mode: params.cot_mode.map(|mode| mode.db_name().to_string()),
        evaluator: params.evaluator,
        git_hash: params.git_hash,
        model_name: params.model_name,
        arch_version: params.arch_version,
        data_version: params.data_version,
        num_params: params.num_params,
        benchmark_name: params.benchmark_name,
        benchmark_names: params
            .field
            .map(benchmark_names_for_field)
            .unwrap_or_default(),
        has_score: params.has_score,
        include_tmp: params.include_tmp.unwrap_or(false),
        include_param_search: params.include_param_search.unwrap_or(false),
    };

    let rows = list_tasks(&state.db, &query)
        .await
        .map_err(ApiError::internal)?;
    let mapped = rows
        .iter()
        .map(to_task_resource)
        .collect::<Result<Vec<_>, _>>()?;
    let (items, has_more) = truncate_has_more(mapped, limit);

    Ok(Json(TaskListResponse {
        items,
        limit,
        offset,
        has_more,
    }))
}

#[utoipa::path(
    get,
    path = "/api/v1/tasks/{task_id}",
    params(("task_id" = i32, Path, description = "Task ID")),
    responses(
        (status = 200, description = "Task detail with derived counters", body = TaskDetailResponse),
        (status = 404, description = "Task not found", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "tasks"
)]
pub(crate) async fn task_detail(
    State(state): State<AppState>,
    Path(task_id): Path<i32>,
) -> ApiResult<TaskDetailResponse> {
    let detail = get_task_detail(&state.db, task_id)
        .await
        .map_err(ApiError::internal)?
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

#[utoipa::path(
    get,
    path = "/api/v1/tasks/{task_id}/attempts",
    params(
        ("task_id" = i32, Path, description = "Task ID"),
        TaskAttemptsParams
    ),
    responses(
        (status = 200, description = "Persisted attempts for a task", body = TaskAttemptsResponse),
        (status = 400, description = "Bad request", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "tasks"
)]
pub(crate) async fn task_attempts(
    State(state): State<AppState>,
    Path(task_id): Path<i32>,
    Query(params): Query<TaskAttemptsParams>,
) -> ApiResult<TaskAttemptsResponse> {
    let (limit, fetch_limit) = resolve_limit(params.limit)?;
    let offset = resolve_offset(params.offset);
    let query = TaskAttemptsQuery {
        limit: fetch_limit,
        offset: i64::from(offset),
        only_failed: params.only_failed.unwrap_or(false),
        has_checker: params.has_checker,
        needs_human_review: params.needs_human_review,
        sample_index: params.sample_index,
    };

    let rows = list_task_attempts(&state.db, task_id, &query)
        .await
        .map_err(ApiError::internal)?;
    let mapped = rows
        .iter()
        .map(to_task_attempt_resource)
        .collect::<Vec<_>>();
    let (items, has_more) = truncate_has_more(mapped, limit);

    Ok(Json(TaskAttemptsResponse {
        items,
        limit,
        offset,
        has_more,
    }))
}

#[utoipa::path(
    get,
    path = "/api/v1/completions/{completions_id}",
    params(("completions_id" = i32, Path, description = "Completion row ID")),
    responses(
        (status = 200, description = "Full completion/eval/checker detail", body = CompletionDetailResponse),
        (status = 404, description = "Completion not found", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "attempts"
)]
pub(crate) async fn completion_detail(
    State(state): State<AppState>,
    Path(completions_id): Path<i32>,
) -> ApiResult<CompletionDetailResponse> {
    let detail = get_completion_detail(&state.db, completions_id)
        .await
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::not_found(format!("completion `{completions_id}` not found")))?;

    Ok(Json(to_completion_detail_response(detail)?))
}

#[utoipa::path(
    get,
    path = "/api/v1/review-queue",
    params(ReviewQueueParams),
    responses(
        (status = 200, description = "Failed attempts flagged for human review", body = ReviewQueueResponse),
        (status = 400, description = "Bad request", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "review"
)]
pub(crate) async fn review_queue(
    State(state): State<AppState>,
    Query(params): Query<ReviewQueueParams>,
) -> ApiResult<ReviewQueueResponse> {
    let (limit, fetch_limit) = resolve_limit(params.limit)?;
    let offset = resolve_offset(params.offset);
    let query = ReviewQueueQuery {
        limit: fetch_limit,
        offset: i64::from(offset),
        model_name: params.model_name,
        benchmark_name: params.benchmark_name,
        benchmark_names: params
            .field
            .map(benchmark_names_for_field)
            .unwrap_or_default(),
        task_id: params.task_id,
    };

    let rows = list_review_queue(&state.db, &query)
        .await
        .map_err(ApiError::internal)?;
    let mapped = rows
        .into_iter()
        .map(to_review_queue_resource)
        .collect::<Result<Vec<_>, _>>()?;
    let (items, has_more) = truncate_has_more(mapped, limit);

    Ok(Json(ReviewQueueResponse {
        items,
        limit,
        offset,
        has_more,
    }))
}

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/start",
    responses(
        (status = 200, description = "Started or reused the current admin evaluation process", body = AdminEvalStatusResponse),
        (status = 400, description = "Bad request", body = super::error::ErrorResponse),
        (status = 401, description = "Unauthorized", body = super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::error::ErrorResponse),
        (status = 409, description = "Conflict", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_start(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    state.eval_controller.start(&state.service_config).await?;
    admin_eval_status_impl(&state).await
}

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/pause",
    responses(
        (status = 200, description = "Requested a soft pause for the active evaluation", body = AdminEvalStatusResponse),
        (status = 401, description = "Unauthorized", body = super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::error::ErrorResponse),
        (status = 404, description = "No active evaluation", body = super::error::ErrorResponse),
        (status = 409, description = "Conflict", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_pause(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    state.eval_controller.pause().await?;
    admin_eval_status_impl(&state).await
}

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/resume",
    responses(
        (status = 200, description = "Resumed a paused evaluation", body = AdminEvalStatusResponse),
        (status = 401, description = "Unauthorized", body = super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::error::ErrorResponse),
        (status = 404, description = "No active evaluation", body = super::error::ErrorResponse),
        (status = 409, description = "Conflict", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_resume(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    state.eval_controller.resume().await?;
    admin_eval_status_impl(&state).await
}

#[utoipa::path(
    post,
    path = "/api/v1/admin/eval/cancel",
    responses(
        (status = 200, description = "Requested a soft cancel for the active evaluation", body = AdminEvalStatusResponse),
        (status = 401, description = "Unauthorized", body = super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::error::ErrorResponse),
        (status = 404, description = "No active evaluation", body = super::error::ErrorResponse),
        (status = 409, description = "Conflict", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_cancel(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    state.eval_controller.cancel().await?;
    admin_eval_status_impl(&state).await
}

#[utoipa::path(
    get,
    path = "/api/v1/admin/eval/status",
    responses(
        (status = 200, description = "Current admin evaluation process status", body = AdminEvalStatusResponse),
        (status = 401, description = "Unauthorized", body = super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_eval_status(
    State(state): State<AppState>,
) -> ApiResult<AdminEvalStatusResponse> {
    admin_eval_status_impl(&state).await
}

#[utoipa::path(
    get,
    path = "/api/v1/admin/health",
    responses(
        (status = 200, description = "Current eval config health fan-out", body = AdminHealthResponse),
        (status = 401, description = "Unauthorized", body = super::error::ErrorResponse),
        (status = 403, description = "Forbidden", body = super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::error::ErrorResponse)
    ),
    tag = "admin"
)]
pub(crate) async fn admin_health(State(state): State<AppState>) -> ApiResult<AdminHealthResponse> {
    let targets = fetch_health_targets(&state.health_client, &state.service_config).await?;
    Ok(Json(AdminHealthResponse {
        targets: targets
            .into_iter()
            .map(to_admin_health_target_resource)
            .collect(),
    }))
}

fn resolve_limit(limit: Option<u32>) -> Result<(u32, i64), ApiError> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT as u32);
    if limit == 0 {
        return Err(ApiError::bad_request("limit must be greater than 0"));
    }
    if i64::from(limit) > MAX_LIMIT {
        return Err(ApiError::bad_request(format!(
            "limit must be <= {MAX_LIMIT}"
        )));
    }
    Ok((limit, i64::from(limit) + 1))
}

fn resolve_offset(offset: Option<u32>) -> u32 {
    offset.unwrap_or(0)
}

fn truncate_has_more<T>(mut items: Vec<T>, limit: u32) -> (Vec<T>, bool) {
    let has_more = items.len() > limit as usize;
    if has_more {
        items.truncate(limit as usize);
    }
    (items, has_more)
}

async fn admin_eval_status_impl(state: &AppState) -> ApiResult<AdminEvalStatusResponse> {
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

    let rows = list_tasks(
        &state.db,
        &TaskListQuery {
            limit: 10_000,
            offset: 0,
            config_path: Some(snapshot.config_path.clone()),
            latest_only: false,
            status: None,
            cot_mode: None,
            evaluator: None,
            git_hash: None,
            model_name: None,
            arch_version: None,
            data_version: None,
            num_params: None,
            benchmark_name: None,
            benchmark_names: Vec::new(),
            has_score: None,
            include_tmp: true,
            include_param_search: true,
        },
    )
    .await
    .map_err(ApiError::internal)?;
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
