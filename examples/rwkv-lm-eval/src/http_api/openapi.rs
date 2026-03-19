use utoipa::OpenApi;

#[allow(unused_imports)]
use super::error::ErrorResponse;
#[allow(unused_imports)]
use super::handlers::{
    __path_benchmarks, __path_completion_detail, __path_health, __path_index, __path_meta,
    __path_models, __path_openapi_json, __path_review_queue, __path_task_attempts,
    __path_task_detail, __path_tasks, benchmarks, completion_detail, health, index, meta, models,
    openapi_json, review_queue, task_attempts, task_detail, tasks,
};
use super::schema::{
    ApiCompletionStatus, ApiCotMode, ApiTaskStatus, BenchmarkField, BenchmarkResource,
    CheckerSummary, CompletionDetailResponse, HealthResponse, IndexResponse, MetaResponse,
    ModelResource, ReviewQueueResource, ReviewQueueResponse, SamplingSummary, ScoreSummary,
    TaskAttemptResource, TaskAttemptsResponse, TaskDetailResponse, TaskListResponse, TaskResource,
};

#[derive(OpenApi)]
#[openapi(
    paths(
        index,
        health,
        openapi_json,
        meta,
        models,
        benchmarks,
        tasks,
        task_detail,
        task_attempts,
        completion_detail,
        review_queue
    ),
    components(
        schemas(
            IndexResponse,
            HealthResponse,
            ErrorResponse,
            MetaResponse,
            ModelResource,
            BenchmarkResource,
            SamplingSummary,
            ScoreSummary,
            TaskResource,
            TaskListResponse,
            TaskDetailResponse,
            CheckerSummary,
            TaskAttemptResource,
            TaskAttemptsResponse,
            CompletionDetailResponse,
            ReviewQueueResource,
            ReviewQueueResponse,
            BenchmarkField,
            ApiTaskStatus,
            ApiCompletionStatus,
            ApiCotMode
        )
    ),
    tags(
        (name = "system", description = "Service discovery and health"),
        (name = "meta", description = "Selectors and metadata"),
        (name = "tasks", description = "Evaluation task views"),
        (name = "attempts", description = "Completion-level drill-down"),
        (name = "review", description = "Human review queue")
    )
)]
pub(crate) struct ApiDoc;
