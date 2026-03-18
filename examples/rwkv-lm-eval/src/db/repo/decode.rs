use sqlx::Row;
use sqlx::postgres::PgRow;

use crate::db::{CheckerRecord, CompletionStatus, TaskRecord, TaskStatus};

pub(crate) fn decode_task_record(row: &PgRow) -> Result<TaskRecord, String> {
    let task_status = row
        .try_get::<String, _>("task_status")
        .map_err(|err| format!("decode task status failed: {err}"))?;

    Ok(TaskRecord {
        task_id: row
            .try_get("task_id")
            .map_err(|err| format!("decode task_id failed: {err}"))?,
        config_path: row
            .try_get("config_path")
            .map_err(|err| format!("decode config_path failed: {err}"))?,
        evaluator: row
            .try_get("evaluator")
            .map_err(|err| format!("decode evaluator failed: {err}"))?,
        is_param_search: row
            .try_get("is_param_search")
            .map_err(|err| format!("decode is_param_search failed: {err}"))?,
        is_tmp: row
            .try_get("is_tmp")
            .map_err(|err| format!("decode is_tmp failed: {err}"))?,
        task_created_at: row
            .try_get("task_created_at")
            .map_err(|err| format!("decode task_created_at failed: {err}"))?,
        task_status: TaskStatus::parse(&task_status)?,
        git_hash: row
            .try_get("git_hash")
            .map_err(|err| format!("decode git_hash failed: {err}"))?,
        task_desc: row
            .try_get("task_desc")
            .map_err(|err| format!("decode task_desc failed: {err}"))?,
        sampling_config_json: row
            .try_get("sampling_config_json")
            .map_err(|err| format!("decode sampling_config_json failed: {err}"))?,
        log_path: row
            .try_get("log_path")
            .map_err(|err| format!("decode log_path failed: {err}"))?,
        model_id: row
            .try_get("model_id")
            .map_err(|err| format!("decode model_id failed: {err}"))?,
        model_name: row
            .try_get("model_name")
            .map_err(|err| format!("decode model_name failed: {err}"))?,
        arch_version: row
            .try_get("arch_version")
            .map_err(|err| format!("decode arch_version failed: {err}"))?,
        data_version: row
            .try_get("data_version")
            .map_err(|err| format!("decode data_version failed: {err}"))?,
        num_params: row
            .try_get("num_params")
            .map_err(|err| format!("decode num_params failed: {err}"))?,
        benchmark_id: row
            .try_get("benchmark_id")
            .map_err(|err| format!("decode benchmark_id failed: {err}"))?,
        benchmark_name: row
            .try_get("benchmark_name")
            .map_err(|err| format!("decode benchmark_name failed: {err}"))?,
        benchmark_split: row
            .try_get("benchmark_split")
            .map_err(|err| format!("decode benchmark_split failed: {err}"))?,
        benchmark_url: row
            .try_get("benchmark_url")
            .map_err(|err| format!("decode benchmark_url failed: {err}"))?,
        benchmark_status: row
            .try_get("benchmark_status")
            .map_err(|err| format!("decode benchmark_status failed: {err}"))?,
        num_samples: row
            .try_get("num_samples")
            .map_err(|err| format!("decode num_samples failed: {err}"))?,
        score_id: row
            .try_get("score_id")
            .map_err(|err| format!("decode score_id failed: {err}"))?,
        score_created_at: row
            .try_get("score_created_at")
            .map_err(|err| format!("decode score_created_at failed: {err}"))?,
        score_cot_mode: row
            .try_get("score_cot_mode")
            .map_err(|err| format!("decode score_cot_mode failed: {err}"))?,
        metrics_json: row
            .try_get("metrics_json")
            .map_err(|err| format!("decode metrics_json failed: {err}"))?,
    })
}

pub(crate) fn decode_completion_status(
    row: &PgRow,
    column: &str,
) -> Result<CompletionStatus, String> {
    let status = row
        .try_get::<String, _>(column)
        .map_err(|err| format!("decode {column} failed: {err}"))?;
    CompletionStatus::parse(&status)
}

pub(crate) fn decode_checker_record(row: &PgRow) -> Result<Option<CheckerRecord>, String> {
    let checker_id = row
        .try_get::<Option<i32>, _>("checker_id")
        .map_err(|err| format!("decode checker_id failed: {err}"))?;

    let Some(checker_id) = checker_id else {
        return Ok(None);
    };

    Ok(Some(CheckerRecord {
        checker_id,
        answer_correct: row
            .try_get("answer_correct")
            .map_err(|err| format!("decode answer_correct failed: {err}"))?,
        instruction_following_error: row
            .try_get("instruction_following_error")
            .map_err(|err| format!("decode instruction_following_error failed: {err}"))?,
        world_knowledge_error: row
            .try_get("world_knowledge_error")
            .map_err(|err| format!("decode world_knowledge_error failed: {err}"))?,
        math_error: row
            .try_get("math_error")
            .map_err(|err| format!("decode math_error failed: {err}"))?,
        reasoning_logic_error: row
            .try_get("reasoning_logic_error")
            .map_err(|err| format!("decode reasoning_logic_error failed: {err}"))?,
        thought_contains_correct_answer: row
            .try_get("thought_contains_correct_answer")
            .map_err(|err| format!("decode thought_contains_correct_answer failed: {err}"))?,
        needs_human_review: row
            .try_get("needs_human_review")
            .map_err(|err| format!("decode needs_human_review failed: {err}"))?,
        reason: row
            .try_get("checker_reason")
            .map_err(|err| format!("decode checker_reason failed: {err}"))?,
        created_at: row
            .try_get("checker_created_at")
            .map_err(|err| format!("decode checker_created_at failed: {err}"))?,
    }))
}
