use axum::{Json, extract::State};

use crate::{
    dtos::ModelResource,
    routes::http_api::{AppState, error::ApiResult},
    services::meta,
};
use super::to_model_resource;

#[utoipa::path(
    get,
    path = "/api/v1/models",
    responses(
        (status = 200, description = "Registered models", body = [ModelResource]),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "meta"
)]
pub(crate) async fn models(State(state): State<AppState>) -> ApiResult<Vec<ModelResource>> {
    let rows = meta::models(&state.db).await?;
    Ok(Json(rows.iter().map(to_model_resource).collect()))
}
