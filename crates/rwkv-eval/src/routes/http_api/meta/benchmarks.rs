use axum::{Json, extract::State};

use crate::{
    dtos::BenchmarkResource,
    routes::http_api::{AppState, error::ApiResult},
    services::meta,
};
use super::to_benchmark_resource;

#[utoipa::path(
    get,
    path = "/api/v1/benchmarks",
    responses(
        (status = 200, description = "Registered benchmarks", body = [BenchmarkResource]),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "meta"
)]
pub(crate) async fn benchmarks(State(state): State<AppState>) -> ApiResult<Vec<BenchmarkResource>> {
    let rows = meta::benchmarks(&state.db).await?;
    Ok(Json(rows.iter().map(to_benchmark_resource).collect()))
}
