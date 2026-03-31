use crate::{
    db::{
        Db, TaskAttemptRecord, TaskAttemptsQuery, TaskDetailRecord, TaskListQuery, TaskRecord,
        get_task_detail, list_task_attempts, list_tasks,
    },
    dtos::{ApiTaskStatus, ListTasksParams, TaskAttemptsParams},
    services::{ServiceError, ServiceResult, meta::benchmark_names_for_field},
};

const DEFAULT_LIMIT: i64 = 50;
const MAX_LIMIT: i64 = 200;

pub struct ListedTasks {
    pub rows: Vec<TaskRecord>,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}

pub struct ListedTaskAttempts {
    pub rows: Vec<TaskAttemptRecord>,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}

pub async fn list(db: &Db, params: ListTasksParams) -> ServiceResult<ListedTasks> {
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
    let rows = list_tasks(db, &query)
        .await
        .map_err(ServiceError::internal)?;
    let (rows, has_more) = truncate_has_more(rows, limit);

    Ok(ListedTasks {
        rows,
        limit,
        offset,
        has_more,
    })
}

pub async fn detail(db: &Db, task_id: i32) -> ServiceResult<Option<TaskDetailRecord>> {
    get_task_detail(db, task_id)
        .await
        .map_err(ServiceError::internal)
}

pub async fn attempts(
    db: &Db,
    task_id: i32,
    params: TaskAttemptsParams,
) -> ServiceResult<ListedTaskAttempts> {
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
    let rows = list_task_attempts(db, task_id, &query)
        .await
        .map_err(ServiceError::internal)?;
    let (rows, has_more) = truncate_has_more(rows, limit);

    Ok(ListedTaskAttempts {
        rows,
        limit,
        offset,
        has_more,
    })
}

pub async fn by_config_path(db: &Db, config_path: String) -> ServiceResult<Vec<TaskRecord>> {
    list_tasks(
        db,
        &TaskListQuery {
            limit: 10_000,
            offset: 0,
            config_path: Some(config_path),
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
    .map_err(ServiceError::internal)
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

#[cfg(test)]
mod tests {
    use super::{resolve_limit, truncate_has_more};

    #[test]
    fn rejects_zero_limit() {
        let err = resolve_limit(Some(0)).unwrap_err();
        assert_eq!(err.kind(), crate::services::ServiceErrorKind::BadRequest);
    }

    #[test]
    fn truncates_extra_items() {
        let (items, has_more) = truncate_has_more(vec![1, 2, 3], 2);
        assert_eq!(items, vec![1, 2]);
        assert!(has_more);
    }
}
