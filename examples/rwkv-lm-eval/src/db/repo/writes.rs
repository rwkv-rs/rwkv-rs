use sqlx::{Postgres, QueryBuilder, query, query_scalar};

use crate::db::CompletionStatus;
use crate::db::{
    BenchmarkInsert, CheckerInsert, CompletionInsert, Db, EvalInsert, ModelInsert, ScoreInsert,
    StartupRecoveryStats, TaskInsert, TaskStatus,
};

pub async fn upsert_model(db: &Db, insert: &ModelInsert) -> Result<i32, String> {
    query_scalar(
        r#"
        INSERT INTO model (data_version, arch_version, num_params, model_name)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (arch_version, data_version, num_params, model_name)
        DO UPDATE SET model_name = EXCLUDED.model_name
        RETURNING model_id
        "#,
    )
    .bind(&insert.data_version)
    .bind(&insert.arch_version)
    .bind(&insert.num_params)
    .bind(&insert.model_name)
    .fetch_one(&db.pool)
    .await
    .map_err(|err| format!("upsert model failed: {err}"))
}

pub async fn upsert_benchmark(db: &Db, insert: &BenchmarkInsert) -> Result<i32, String> {
    query_scalar(
        r#"
        INSERT INTO benchmark (benchmark_name, benchmark_split, url, status, num_samples)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (benchmark_name, benchmark_split)
        DO UPDATE
        SET url = EXCLUDED.url,
            status = EXCLUDED.status,
            num_samples = EXCLUDED.num_samples
        RETURNING benchmark_id
        "#,
    )
    .bind(&insert.benchmark_name)
    .bind(&insert.benchmark_split)
    .bind(&insert.url)
    .bind(&insert.status)
    .bind(insert.num_samples)
    .fetch_one(&db.pool)
    .await
    .map_err(|err| format!("upsert benchmark failed: {err}"))
}

pub async fn insert_task(db: &Db, insert: &TaskInsert) -> Result<i32, String> {
    query_scalar(
        r#"
        INSERT INTO task (
            config_path,
            evaluator,
            is_param_search,
            is_tmp,
            created_at,
            status,
            git_hash,
            model_id,
            benchmark_id,
            "desc",
            sampling_config,
            log_path
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            CURRENT_TIMESTAMP,
            $5,
            $6,
            $7,
            $8,
            $9,
            $10::jsonb,
            $11
        )
        RETURNING task_id
        "#,
    )
    .bind(&insert.config_path)
    .bind(&insert.evaluator)
    .bind(insert.is_param_search)
    .bind(insert.is_tmp)
    .bind(insert.status.as_str())
    .bind(&insert.git_hash)
    .bind(insert.model_id)
    .bind(insert.benchmark_id)
    .bind(&insert.desc)
    .bind(&insert.sampling_config_json)
    .bind(&insert.log_path)
    .fetch_one(&db.pool)
    .await
    .map_err(|err| format!("insert task failed: {err}"))
}

pub async fn insert_completion_and_eval(
    db: &Db,
    completion: &CompletionInsert,
    eval: &EvalInsert,
) -> Result<i32, String> {
    let mut tx = db
        .pool
        .begin()
        .await
        .map_err(|err| format!("begin completion/eval transaction failed: {err}"))?;

    let completions_id: i32 = query_scalar(
        r#"
        INSERT INTO completions (
            task_id,
            context,
            sample_index,
            avg_repeat_index,
            pass_index,
            created_at,
            status
        )
        VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, $6)
        RETURNING completions_id
        "#,
    )
    .bind(completion.task_id)
    .bind(&completion.context)
    .bind(completion.sample_index)
    .bind(completion.avg_repeat_index)
    .bind(completion.pass_index)
    .bind(completion.status.as_str())
    .fetch_one(&mut *tx)
    .await
    .map_err(|err| format!("insert completion failed: {err}"))?;

    query(
        r#"
        INSERT INTO eval (
            completions_id,
            answer,
            ref_answer,
            is_passed,
            fail_reason,
            created_at
        )
        VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
        "#,
    )
    .bind(completions_id)
    .bind(&eval.answer)
    .bind(&eval.ref_answer)
    .bind(eval.is_passed)
    .bind(&eval.fail_reason)
    .execute(&mut *tx)
    .await
    .map_err(|err| format!("insert eval failed: {err}"))?;

    tx.commit()
        .await
        .map_err(|err| format!("commit completion/eval transaction failed: {err}"))?;

    Ok(completions_id)
}

pub async fn insert_checker(db: &Db, insert: &CheckerInsert) -> Result<(), String> {
    query(
        r#"
        INSERT INTO checker (
            completions_id,
            answer_correct,
            instruction_following_error,
            world_knowledge_error,
            math_error,
            reasoning_logic_error,
            thought_contains_correct_answer,
            needs_human_review,
            reason,
            created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
        "#,
    )
    .bind(insert.completions_id)
    .bind(insert.answer_correct)
    .bind(insert.instruction_following_error)
    .bind(insert.world_knowledge_error)
    .bind(insert.math_error)
    .bind(insert.reasoning_logic_error)
    .bind(insert.thought_contains_correct_answer)
    .bind(insert.needs_human_review)
    .bind(&insert.reason)
    .execute(&db.pool)
    .await
    .map_err(|err| format!("insert checker failed: {err}"))?;

    Ok(())
}

pub async fn delete_score_by_task_id(db: &Db, task_id: i32) -> Result<(), String> {
    query("DELETE FROM scores WHERE task_id = $1")
        .bind(task_id)
        .execute(&db.pool)
        .await
        .map_err(|err| format!("delete task score failed: {err}"))?;

    Ok(())
}

pub async fn insert_score(db: &Db, insert: &ScoreInsert) -> Result<(), String> {
    query(
        r#"
        INSERT INTO scores (task_id, cot_mode, metrics, created_at)
        VALUES ($1, $2, $3::jsonb, CURRENT_TIMESTAMP)
        "#,
    )
    .bind(insert.task_id)
    .bind(&insert.cot_mode)
    .bind(&insert.metrics_json)
    .execute(&db.pool)
    .await
    .map_err(|err| format!("insert score failed: {err}"))?;

    Ok(())
}

pub async fn update_task_status(db: &Db, task_id: i32, status: TaskStatus) -> Result<(), String> {
    query("UPDATE task SET status = $1 WHERE task_id = $2")
        .bind(status.as_str())
        .bind(task_id)
        .execute(&db.pool)
        .await
        .map_err(|err| format!("update task status failed: {err}"))?;

    Ok(())
}

pub async fn recover_running_tasks(db: &Db) -> Result<StartupRecoveryStats, String> {
    let mut tx = db
        .pool
        .begin()
        .await
        .map_err(|err| format!("begin startup recovery transaction failed: {err}"))?;

    let task_ids: Vec<i32> = query_scalar(
        r#"
        UPDATE task
        SET status = $1
        WHERE status = $2
        RETURNING task_id
        "#,
    )
    .bind(TaskStatus::Failed.as_str())
    .bind(TaskStatus::Running.as_str())
    .fetch_all(&mut *tx)
    .await
    .map_err(|err| format!("mark running tasks as failed failed: {err}"))?;
    let failed_completion_count = if task_ids.is_empty() {
        0
    } else {
        let mut builder = QueryBuilder::<Postgres>::new("UPDATE completions SET status = ");
        builder
            .push_bind(CompletionStatus::Failed.as_str())
            .push(" WHERE status = ")
            .push_bind(CompletionStatus::Running.as_str())
            .push(" AND task_id IN (");

        let mut separated = builder.separated(", ");
        for task_id in &task_ids {
            separated.push_bind(task_id);
        }
        separated.push_unseparated(")");

        builder
            .build()
            .execute(&mut *tx)
            .await
            .map_err(|err| format!("mark running completions as failed failed: {err}"))?
            .rows_affected()
    };

    tx.commit()
        .await
        .map_err(|err| format!("commit startup recovery transaction failed: {err}"))?;

    Ok(StartupRecoveryStats {
        failed_task_count: task_ids.len() as u64,
        failed_completion_count,
    })
}
