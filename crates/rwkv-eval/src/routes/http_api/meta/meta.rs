use axum::{Json, extract::State};

use crate::{
    dtos::{ApiCompletionStatus, ApiCotMode, ApiTaskStatus, BenchmarkField, MetaResponse},
    routes::http_api::{AppState, error::ApiResult},
    services::meta,
};
use super::{to_benchmark_resource, to_model_resource};

#[utoipa::path(
    get,
    path = "/api/v1/meta",
    responses(
        (status = 200, description = "Dashboard metadata", body = MetaResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "meta"
)]
pub(crate) async fn meta(State(state): State<AppState>) -> ApiResult<MetaResponse> {
    let (model_rows, benchmark_rows) = meta::meta(&state.db).await?;

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
