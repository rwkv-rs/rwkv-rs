use utoipa::OpenApi;

#[allow(unused_imports)]
use super::error::ErrorResponse;
#[allow(unused_imports)]
use super::{
    admin::{
        __path_admin_eval_cancel,
        __path_admin_eval_draft,
        __path_admin_eval_pause,
        __path_admin_eval_resume,
        __path_admin_eval_start,
        __path_admin_eval_status,
        __path_admin_health,
        admin_eval_cancel,
        admin_eval_draft,
        admin_eval_pause,
        admin_eval_resume,
        admin_eval_start,
        admin_eval_status,
        admin_health,
    },
    completions::{__path_completion_detail, completion_detail},
    meta::{__path_benchmarks, __path_meta, __path_models, benchmarks, meta, models},
    review_queue::{__path_review_queue, review_queue},
    system::{__path_health, __path_index, __path_openapi_json, health, index, openapi_json},
    tasks::{
        __path_task_attempts,
        __path_task_detail,
        __path_tasks,
        task_attempts,
        task_detail,
        tasks,
    },
};
use crate::dtos::{
    AdminDependencyResource,
    AdminEvalConfigDto,
    AdminEvalStatusResponse,
    AdminHealthResponse,
    AdminHealthTargetResource,
    ApiCompletionStatus,
    ApiCotMode,
    ApiTaskStatus,
    BenchmarkField,
    BenchmarkResource,
    CheckerSummary,
    CompletionDetailResponse,
    HealthResponse,
    IndexResponse,
    MetaResponse,
    ModelResource,
    ReviewQueueResource,
    ReviewQueueResponse,
    SamplingSummary,
    ScoreSummary,
    TaskAttemptResource,
    TaskAttemptsResponse,
    TaskDetailResponse,
    TaskListResponse,
    TaskProgress,
    TaskResource,
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
        review_queue,
        admin_eval_draft,
        admin_eval_start,
        admin_eval_pause,
        admin_eval_resume,
        admin_eval_cancel,
        admin_eval_status,
        admin_health
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
            TaskProgress,
            TaskResource,
            TaskListResponse,
            TaskDetailResponse,
            CheckerSummary,
            TaskAttemptResource,
            TaskAttemptsResponse,
            CompletionDetailResponse,
            ReviewQueueResource,
            ReviewQueueResponse,
            AdminDependencyResource,
            AdminEvalConfigDto,
            AdminEvalStatusResponse,
            AdminHealthTargetResource,
            AdminHealthResponse,
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
        (name = "review", description = "Human review queue"),
        (name = "admin", description = "Evaluation control and aggregated health")
    )
)]
pub(crate) struct ApiDoc;

#[cfg(test)]
mod tests {
    use utoipa::OpenApi;

    use super::ApiDoc;

    #[test]
    fn openapi_contains_all_route_groups() {
        let json = ApiDoc::openapi().to_pretty_json().unwrap();
        for path in [
            "/",
            "/health",
            "/openapi.json",
            "/api/v1/meta",
            "/api/v1/models",
            "/api/v1/benchmarks",
            "/api/v1/tasks",
            "/api/v1/tasks/{task_id}",
            "/api/v1/tasks/{task_id}/attempts",
            "/api/v1/completions/{completions_id}",
            "/api/v1/review-queue",
            "/api/v1/admin/eval/draft",
            "/api/v1/admin/eval/start",
            "/api/v1/admin/eval/pause",
            "/api/v1/admin/eval/resume",
            "/api/v1/admin/eval/cancel",
            "/api/v1/admin/eval/status",
            "/api/v1/admin/health",
        ] {
            assert!(json.contains(path), "missing path {path}");
        }
    }
}
