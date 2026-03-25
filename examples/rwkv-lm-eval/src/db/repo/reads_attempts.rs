use sqlx::{Postgres, QueryBuilder, Row, query};

use super::decode::{decode_checker_record, decode_completion_status, decode_task_record};
use crate::db::{
    CompletionDetailRecord,
    Db,
    ReviewQueueQuery,
    ReviewQueueRecord,
    TaskAttemptRecord,
    TaskAttemptsQuery,
};

pub async fn list_task_attempts(
    db: &Db,
    task_id: i32,
    filter: &TaskAttemptsQuery,
) -> Result<Vec<TaskAttemptRecord>, String> {
    let mut builder = QueryBuilder::<Postgres>::new(
        r#"
        SELECT
            c.completions_id,
            c.task_id,
            c.sample_index,
            c.avg_repeat_index,
            c.pass_index,
            c.status AS completion_status,
            c.created_at::text AS completion_created_at,
            e.answer,
            e.ref_answer,
            e.is_passed,
            e.fail_reason,
            e.created_at::text AS eval_created_at,
            CASE
                WHEN char_length(c.context) > 240 THEN substring(c.context from 1 for 240) || '…'
                ELSE c.context
            END AS context_preview,
            ch.checker_id,
            ch.answer_correct,
            ch.instruction_following_error,
            ch.world_knowledge_error,
            ch.math_error,
            ch.reasoning_logic_error,
            ch.thought_contains_correct_answer,
            ch.needs_human_review,
            ch.reason AS checker_reason,
            ch.created_at::text AS checker_created_at
        FROM completions c
        JOIN eval e ON e.completions_id = c.completions_id
        LEFT JOIN checker ch ON ch.completions_id = c.completions_id
        WHERE c.task_id =
        "#,
    );
    builder.push_bind(task_id);

    if filter.only_failed {
        builder.push(" AND e.is_passed = FALSE");
    }
    if let Some(has_checker) = filter.has_checker {
        if has_checker {
            builder.push(" AND ch.checker_id IS NOT NULL");
        } else {
            builder.push(" AND ch.checker_id IS NULL");
        }
    }
    if let Some(needs_human_review) = filter.needs_human_review {
        builder
            .push(" AND COALESCE(ch.needs_human_review, FALSE) = ")
            .push_bind(needs_human_review);
    }
    if let Some(sample_index) = filter.sample_index {
        builder
            .push(" AND c.sample_index = ")
            .push_bind(sample_index);
    }

    builder.push(
        r#"
        ORDER BY c.avg_repeat_index, c.sample_index, c.pass_index
        LIMIT
        "#,
    );
    builder
        .push_bind(filter.limit)
        .push(" OFFSET ")
        .push_bind(filter.offset);

    let rows = builder
        .build()
        .fetch_all(&db.pool)
        .await
        .map_err(|err| format!("list task attempts failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            Ok(TaskAttemptRecord {
                completions_id: row
                    .try_get("completions_id")
                    .map_err(|err| format!("decode completions_id failed: {err}"))?,
                task_id: row
                    .try_get("task_id")
                    .map_err(|err| format!("decode task_id failed: {err}"))?,
                sample_index: row
                    .try_get("sample_index")
                    .map_err(|err| format!("decode sample_index failed: {err}"))?,
                avg_repeat_index: row
                    .try_get("avg_repeat_index")
                    .map_err(|err| format!("decode avg_repeat_index failed: {err}"))?,
                pass_index: row
                    .try_get("pass_index")
                    .map_err(|err| format!("decode pass_index failed: {err}"))?,
                completion_status: decode_completion_status(&row, "completion_status")?,
                completion_created_at: row
                    .try_get("completion_created_at")
                    .map_err(|err| format!("decode completion_created_at failed: {err}"))?,
                answer: row
                    .try_get("answer")
                    .map_err(|err| format!("decode answer failed: {err}"))?,
                ref_answer: row
                    .try_get("ref_answer")
                    .map_err(|err| format!("decode ref_answer failed: {err}"))?,
                is_passed: row
                    .try_get("is_passed")
                    .map_err(|err| format!("decode is_passed failed: {err}"))?,
                fail_reason: row
                    .try_get("fail_reason")
                    .map_err(|err| format!("decode fail_reason failed: {err}"))?,
                eval_created_at: row
                    .try_get("eval_created_at")
                    .map_err(|err| format!("decode eval_created_at failed: {err}"))?,
                context_preview: row
                    .try_get("context_preview")
                    .map_err(|err| format!("decode context_preview failed: {err}"))?,
                checker: decode_checker_record(&row)?,
            })
        })
        .collect()
}

pub async fn get_completion_detail(
    db: &Db,
    completions_id: i32,
) -> Result<Option<CompletionDetailRecord>, String> {
    let row = query(
        r#"
        SELECT
            c.completions_id,
            c.sample_index,
            c.avg_repeat_index,
            c.pass_index,
            c.status AS completion_status,
            c.created_at::text AS completion_created_at,
            c.context,
            e.answer,
            e.ref_answer,
            e.is_passed,
            e.fail_reason,
            e.created_at::text AS eval_created_at,
            ch.checker_id,
            ch.answer_correct,
            ch.instruction_following_error,
            ch.world_knowledge_error,
            ch.math_error,
            ch.reasoning_logic_error,
            ch.thought_contains_correct_answer,
            ch.needs_human_review,
            ch.reason AS checker_reason,
            ch.created_at::text AS checker_created_at,
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
            s.metrics::text AS metrics_json
        FROM completions c
        JOIN eval e ON e.completions_id = c.completions_id
        JOIN task t ON t.task_id = c.task_id
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        LEFT JOIN checker ch ON ch.completions_id = c.completions_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE c.completions_id = $1
        "#,
    )
    .bind(completions_id)
    .fetch_optional(&db.pool)
    .await
    .map_err(|err| format!("get completion detail failed: {err}"))?;

    let Some(row) = row else {
        return Ok(None);
    };

    Ok(Some(CompletionDetailRecord {
        task: decode_task_record(&row)?,
        completions_id: row
            .try_get("completions_id")
            .map_err(|err| format!("decode completions_id failed: {err}"))?,
        sample_index: row
            .try_get("sample_index")
            .map_err(|err| format!("decode sample_index failed: {err}"))?,
        avg_repeat_index: row
            .try_get("avg_repeat_index")
            .map_err(|err| format!("decode avg_repeat_index failed: {err}"))?,
        pass_index: row
            .try_get("pass_index")
            .map_err(|err| format!("decode pass_index failed: {err}"))?,
        completion_status: decode_completion_status(&row, "completion_status")?,
        completion_created_at: row
            .try_get("completion_created_at")
            .map_err(|err| format!("decode completion_created_at failed: {err}"))?,
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
        fail_reason: row
            .try_get("fail_reason")
            .map_err(|err| format!("decode fail_reason failed: {err}"))?,
        eval_created_at: row
            .try_get("eval_created_at")
            .map_err(|err| format!("decode eval_created_at failed: {err}"))?,
        checker: decode_checker_record(&row)?,
    }))
}

pub async fn list_review_queue(
    db: &Db,
    filter: &ReviewQueueQuery,
) -> Result<Vec<ReviewQueueRecord>, String> {
    let mut builder = QueryBuilder::<Postgres>::new(
        r#"
        SELECT
            c.completions_id,
            c.sample_index,
            c.avg_repeat_index,
            c.pass_index,
            e.answer,
            e.ref_answer,
            e.fail_reason,
            CASE
                WHEN char_length(c.context) > 240 THEN substring(c.context from 1 for 240) || '…'
                ELSE c.context
            END AS context_preview,
            ch.checker_id,
            ch.answer_correct,
            ch.instruction_following_error,
            ch.world_knowledge_error,
            ch.math_error,
            ch.reasoning_logic_error,
            ch.thought_contains_correct_answer,
            ch.needs_human_review,
            ch.reason AS checker_reason,
            ch.created_at::text AS checker_created_at,
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
            s.metrics::text AS metrics_json
        FROM checker ch
        JOIN completions c ON c.completions_id = ch.completions_id
        JOIN eval e ON e.completions_id = c.completions_id
        JOIN task t ON t.task_id = c.task_id
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE ch.needs_human_review = TRUE
        "#,
    );

    if let Some(task_id) = filter.task_id {
        builder.push(" AND t.task_id = ").push_bind(task_id);
    }
    if let Some(model_name) = filter.model_name.as_ref() {
        builder.push(" AND m.model_name = ").push_bind(model_name);
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

    builder.push(
        r#"
        ORDER BY ch.created_at DESC, ch.checker_id DESC
        LIMIT
        "#,
    );
    builder
        .push_bind(filter.limit)
        .push(" OFFSET ")
        .push_bind(filter.offset);

    let rows = builder
        .build()
        .fetch_all(&db.pool)
        .await
        .map_err(|err| format!("list review queue failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            let checker = decode_checker_record(&row)?
                .ok_or_else(|| "review queue row missing checker data".to_string())?;

            Ok(ReviewQueueRecord {
                task: decode_task_record(&row)?,
                completions_id: row
                    .try_get("completions_id")
                    .map_err(|err| format!("decode completions_id failed: {err}"))?,
                sample_index: row
                    .try_get("sample_index")
                    .map_err(|err| format!("decode sample_index failed: {err}"))?,
                avg_repeat_index: row
                    .try_get("avg_repeat_index")
                    .map_err(|err| format!("decode avg_repeat_index failed: {err}"))?,
                pass_index: row
                    .try_get("pass_index")
                    .map_err(|err| format!("decode pass_index failed: {err}"))?,
                answer: row
                    .try_get("answer")
                    .map_err(|err| format!("decode answer failed: {err}"))?,
                ref_answer: row
                    .try_get("ref_answer")
                    .map_err(|err| format!("decode ref_answer failed: {err}"))?,
                fail_reason: row
                    .try_get("fail_reason")
                    .map_err(|err| format!("decode fail_reason failed: {err}"))?,
                context_preview: row
                    .try_get("context_preview")
                    .map_err(|err| format!("decode context_preview failed: {err}"))?,
                checker,
            })
        })
        .collect()
}
