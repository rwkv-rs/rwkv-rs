use axum::{
    Json,
    extract::{Query, State},
};

use crate::{
    dtos::{ReviewQueueParams, ReviewQueueResponse},
    routes::http_api::{AppState, error::ApiResult},
    services::review_queue,
};
use super::mapper::to_review_queue_resource;

#[utoipa::path(
    get,
    path = "/api/v1/review-queue",
    params(ReviewQueueParams),
    responses(
        (status = 200, description = "Failed attempts flagged for human review", body = ReviewQueueResponse),
        (status = 400, description = "Bad request", body = super::super::error::ErrorResponse),
        (status = 500, description = "Internal server error", body = super::super::error::ErrorResponse)
    ),
    tag = "review"
)]
pub(crate) async fn review_queue(
    State(state): State<AppState>,
    Query(params): Query<ReviewQueueParams>,
) -> ApiResult<ReviewQueueResponse> {
    let listing = review_queue::list(&state.db, params).await?;
    let items = listing
        .rows
        .into_iter()
        .map(to_review_queue_resource)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Json(ReviewQueueResponse {
        items,
        limit: listing.limit,
        offset: listing.offset,
        has_more: listing.has_more,
    }))
}
