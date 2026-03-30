use crate::{
    db::{Db, ReviewQueueQuery, ReviewQueueRecord, list_review_queue},
    dtos::ReviewQueueParams,
    services::{ServiceError, ServiceResult, meta::benchmark_names_for_field},
};

const DEFAULT_LIMIT: i64 = 50;
const MAX_LIMIT: i64 = 200;

pub struct ListedReviewQueue {
    pub rows: Vec<ReviewQueueRecord>,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}

pub async fn list(db: &Db, params: ReviewQueueParams) -> ServiceResult<ListedReviewQueue> {
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

    let rows = list_review_queue(db, &query)
        .await
        .map_err(ServiceError::internal)?;
    let (rows, has_more) = truncate_has_more(rows, limit);

    Ok(ListedReviewQueue {
        rows,
        limit,
        offset,
        has_more,
    })
}

fn resolve_limit(limit: Option<u32>) -> ServiceResult<(u32, i64)> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT as u32);
    if limit == 0 {
        return Err(ServiceError::bad_request("limit must be greater than 0"));
    }
    if i64::from(limit) > MAX_LIMIT {
        return Err(ServiceError::bad_request(format!(
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
