use sqlx::{Postgres, QueryBuilder, Row, query};

use super::decode::decode_task_record;
use crate::db::{
    AttemptRecord,
    Db,
    TaskDetailRecord,
    TaskIdentity,
    TaskListQuery,
    TaskLookup,
    TaskRecord,
    TaskStatus,
};

pub async fn find_tasks_by_identity(
    db: &Db,
    identity: &TaskIdentity,
) -> Result<Vec<TaskLookup>, String> {
    let rows = query(
        r#"
        SELECT task_id, status
        FROM task
        WHERE config_path IS NOT DISTINCT FROM $1
          AND evaluator = $2
          AND git_hash = $3
          AND model_id = $4
          AND benchmark_id = $5
          AND sampling_config = $6::jsonb
        ORDER BY task_id
        "#,
    )
    .bind(&identity.config_path)
    .bind(&identity.evaluator)
    .bind(&identity.git_hash)
    .bind(identity.model_id)
    .bind(identity.benchmark_id)
    .bind(&identity.sampling_config_json)
    .fetch_all(&db.pool)
    .await
    .map_err(|err| format!("find task by identity failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            let status = row
                .try_get::<String, _>("status")
                .map_err(|err| format!("decode task status failed: {err}"))?;
            Ok(TaskLookup {
                task_id: row
                    .try_get("task_id")
                    .map_err(|err| format!("decode task id failed: {err}"))?,
                status: TaskStatus::parse(&status)?,
            })
        })
        .collect()
}

pub async fn list_attempt_records(db: &Db, task_id: i32) -> Result<Vec<AttemptRecord>, String> {
    let rows = query(
        r#"
        SELECT
            c.completions_id,
            c.context,
            c.sample_index,
            c.avg_repeat_index,
            c.pass_index,
            e.answer,
            e.ref_answer,
            e.is_passed,
            ch.checker_id IS NOT NULL AS has_checker
        FROM completions c
        JOIN eval e ON e.completions_id = c.completions_id
        LEFT JOIN checker ch ON ch.completions_id = c.completions_id
        WHERE c.task_id = $1
        ORDER BY c.avg_repeat_index, c.sample_index, c.pass_index
        "#,
    )
    .bind(task_id)
    .fetch_all(&db.pool)
    .await
    .map_err(|err| format!("list attempt records failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            Ok(AttemptRecord {
                completions_id: row
                    .try_get("completions_id")
                    .map_err(|err| format!("decode completions_id failed: {err}"))?,
                key: crate::db::CompletionKey {
                    sample_index: row
                        .try_get("sample_index")
                        .map_err(|err| format!("decode sample_index failed: {err}"))?,
                    avg_repeat_index: row
                        .try_get("avg_repeat_index")
                        .map_err(|err| format!("decode avg_repeat_index failed: {err}"))?,
                    pass_index: row
                        .try_get("pass_index")
                        .map_err(|err| format!("decode pass_index failed: {err}"))?,
                },
                context: row
                    .try_get("context")
                    .map_err(|err| format!("decode context failed: {err}"))?,
                answer: row
                    .try_get("answer")
                    .map_err(|err| format!("decode answer failed: {err}"))?,
                ref_answer: row
                    .try_get("ref_answer")
                    .map_err(|err| format!("decode ref_answer failed: {err}"))?,
                is_passed: row
                    .try_get("is_passed")
                    .map_err(|err| format!("decode is_passed failed: {err}"))?,
                has_checker: row
                    .try_get("has_checker")
                    .map_err(|err| format!("decode has_checker failed: {err}"))?,
            })
        })
        .collect()
}

pub async fn list_tasks(db: &Db, filter: &TaskListQuery) -> Result<Vec<TaskRecord>, String> {
    let mut builder = QueryBuilder::<Postgres>::new(
        r#"
        WITH base AS (
            SELECT
                t.task_id,
                t.config_path,
                t.evaluator,
                t.is_param_search,
                t.is_tmp,
                t.created_at::text AS task_created_at,
                t.created_at AS task_created_at_ts,
                t.status AS task_status,
                t.git_hash,
                t."desc" AS task_desc,
                t.sampling_config::text AS sampling_config_json,
                t.log_path,
                m.model_id,
                m.model_name,
                m.arch_version,
                m.data_version,
                m.num_params,
                b.benchmark_id,
                b.benchmark_name,
                b.benchmark_split,
                b.url AS benchmark_url,
                b.status AS benchmark_status,
                b.num_samples,
                s.score_id,
                s.created_at::text AS score_created_at,
                s.cot_mode AS score_cot_mode,
                s.metrics::text AS metrics_json
            FROM task t
            JOIN model m ON m.model_id = t.model_id
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            LEFT JOIN scores s ON s.task_id = t.task_id
            WHERE 1 = 1
        "#,
    );

    if let Some(status) = filter.status {
        builder.push(" AND t.status = ").push_bind(status.as_str());
    }
    if let Some(cot_mode) = filter.cot_mode.as_ref() {
        builder
            .push(" AND t.sampling_config ->> 'cot_mode' = ")
            .push_bind(cot_mode);
    }
    if let Some(evaluator) = filter.evaluator.as_ref() {
        builder.push(" AND t.evaluator = ").push_bind(evaluator);
    }
    if let Some(git_hash) = filter.git_hash.as_ref() {
        builder.push(" AND t.git_hash = ").push_bind(git_hash);
    }
    if let Some(model_name) = filter.model_name.as_ref() {
        builder.push(" AND m.model_name = ").push_bind(model_name);
    }
    if let Some(arch_version) = filter.arch_version.as_ref() {
        builder
            .push(" AND m.arch_version = ")
            .push_bind(arch_version);
    }
    if let Some(data_version) = filter.data_version.as_ref() {
        builder
            .push(" AND m.data_version = ")
            .push_bind(data_version);
    }
    if let Some(num_params) = filter.num_params.as_ref() {
        builder.push(" AND m.num_params = ").push_bind(num_params);
    }
    if let Some(benchmark_name) = filter.benchmark_name.as_ref() {
        builder
            .push(" AND b.benchmark_name = ")
            .push_bind(benchmark_name);
    }
    if !filter.benchmark_names.is_empty() {
        builder.push(" AND b.benchmark_name IN (");
        let mut separated = builder.separated(", ");
        for benchmark_name in &filter.benchmark_names {
            separated.push_bind(benchmark_name);
        }
        separated.push_unseparated(")");
    }
    if let Some(has_score) = filter.has_score {
        if has_score {
            builder.push(" AND s.score_id IS NOT NULL");
        } else {
            builder.push(" AND s.score_id IS NULL");
        }
    }
    if !filter.include_tmp {
        builder.push(" AND t.is_tmp = FALSE");
    }
    if !filter.include_param_search {
        builder.push(" AND t.is_param_search = FALSE");
    }

    builder.push("\n)\n");

    if filter.latest_only {
        builder.push(
            r#"
            SELECT *
            FROM (
                SELECT DISTINCT ON (model_id, benchmark_id, sampling_config_json) *
                FROM base
                ORDER BY model_id, benchmark_id, sampling_config_json, task_created_at_ts DESC, task_id DESC
            ) latest
            ORDER BY task_created_at_ts DESC, task_id DESC
            "#,
        );
    } else {
        builder.push(
            r#"
            SELECT *
            FROM base
            ORDER BY task_created_at_ts DESC, task_id DESC
            "#,
        );
    }

    builder
        .push(" LIMIT ")
        .push_bind(filter.limit)
        .push(" OFFSET ")
        .push_bind(filter.offset);

    let rows = builder
        .build()
        .fetch_all(&db.pool)
        .await
        .map_err(|err| format!("list tasks failed: {err}"))?;

    rows.into_iter()
        .map(|row| decode_task_record(&row))
        .collect()
}

pub async fn get_task_detail(db: &Db, task_id: i32) -> Result<Option<TaskDetailRecord>, String> {
    let row = query(
        r#"
        SELECT
            t.task_id,
            t.config_path,
            t.evaluator,
            t.is_param_search,
            t.is_tmp,
            t.created_at::text AS task_created_at,
            t.status AS task_status,
            t.git_hash,
            t."desc" AS task_desc,
            t.sampling_config::text AS sampling_config_json,
            t.log_path,
            m.model_id,
            m.model_name,
            m.arch_version,
            m.data_version,
            m.num_params,
            b.benchmark_id,
            b.benchmark_name,
            b.benchmark_split,
            b.url AS benchmark_url,
            b.status AS benchmark_status,
            b.num_samples,
            s.score_id,
            s.created_at::text AS score_created_at,
            s.cot_mode AS score_cot_mode,
            s.metrics::text AS metrics_json,
            COUNT(c.completions_id) AS attempts_total,
            COUNT(*) FILTER (WHERE e.is_passed IS TRUE) AS attempts_passed,
            COUNT(*) FILTER (WHERE e.is_passed IS FALSE) AS attempts_failed,
            COUNT(ch.checker_id) AS attempts_with_checker,
            COUNT(*) FILTER (
                WHERE c.completions_id IS NOT NULL
                  AND e.is_passed IS FALSE
                  AND ch.checker_id IS NULL
            ) AS attempts_missing_checker,
            COUNT(*) FILTER (WHERE ch.needs_human_review IS TRUE) AS attempts_needing_human_review
        FROM task t
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        LEFT JOIN completions c ON c.task_id = t.task_id
        LEFT JOIN eval e ON e.completions_id = c.completions_id
        LEFT JOIN checker ch ON ch.completions_id = c.completions_id
        WHERE t.task_id = $1
        GROUP BY
            t.task_id,
            t.config_path,
            t.evaluator,
            t.is_param_search,
            t.is_tmp,
            t.created_at,
            t.status,
            t.git_hash,
            t."desc",
            t.sampling_config,
            t.log_path,
            m.model_id,
            m.model_name,
            m.arch_version,
            m.data_version,
            m.num_params,
            b.benchmark_id,
            b.benchmark_name,
            b.benchmark_split,
            b.url,
            b.status,
            b.num_samples,
            s.score_id,
            s.created_at,
            s.cot_mode,
            s.metrics
        "#,
    )
    .bind(task_id)
    .fetch_optional(&db.pool)
    .await
    .map_err(|err| format!("get task detail failed: {err}"))?;

    let Some(row) = row else {
        return Ok(None);
    };

    Ok(Some(TaskDetailRecord {
        task: decode_task_record(&row)?,
        attempts_total: row
            .try_get("attempts_total")
            .map_err(|err| format!("decode attempts_total failed: {err}"))?,
        attempts_passed: row
            .try_get("attempts_passed")
            .map_err(|err| format!("decode attempts_passed failed: {err}"))?,
        attempts_failed: row
            .try_get("attempts_failed")
            .map_err(|err| format!("decode attempts_failed failed: {err}"))?,
        attempts_with_checker: row
            .try_get("attempts_with_checker")
            .map_err(|err| format!("decode attempts_with_checker failed: {err}"))?,
        attempts_missing_checker: row
            .try_get("attempts_missing_checker")
            .map_err(|err| format!("decode attempts_missing_checker failed: {err}"))?,
        attempts_needing_human_review: row
            .try_get("attempts_needing_human_review")
            .map_err(|err| format!("decode attempts_needing_human_review failed: {err}"))?,
    }))
}
