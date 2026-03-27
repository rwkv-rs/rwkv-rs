use std::sync::Arc;

use async_openai::{Client, config::OpenAIConfig};
use rwkv_eval::{
    checkers::run_checker,
    datasets::{Benchmark, CoTMode},
};
use tokio::task::JoinSet;

use crate::db::{
    CheckerInsert,
    CompletionInsert,
    CompletionStatus,
    Db,
    EvalInsert,
    insert_checker,
    insert_completion_and_eval,
};
use super::{
    client::ClientWithConfig,
    runtime::{AttemptKey, AttemptOutcome, CheckerRuntime, PendingChecker},
    sampling::AvgKExecutionPlan,
};

pub(crate) async fn execute_attempt(
    benchmark: Arc<dyn Benchmark>,
    target_model: Arc<ClientWithConfig>,
    judger_model_name: Option<String>,
    judger_client: Option<Arc<Client<OpenAIConfig>>>,
    checker_runtime: Option<Arc<CheckerRuntime>>,
    cot_mode: CoTMode,
    n_shot: u8,
    db: Option<Db>,
    task_id: Option<i32>,
    key: AttemptKey,
) -> Result<AttemptOutcome, String> {
    let record = benchmark
        .answer_and_judge(
            &target_model.api_cfg.model,
            &target_model.client,
            judger_model_name.as_deref(),
            judger_client.as_deref(),
            cot_mode,
            n_shot,
            key.sample_index,
        )
        .await;

    if let (Some(db), Some(task_id)) = (db.as_ref(), task_id) {
        let completions_id = insert_completion_and_eval(
            db,
            &CompletionInsert {
                task_id,
                context: record.context.clone(),
                sample_index: key.sample_index as i32,
                avg_repeat_index: key.avg_repeat_index as i32,
                pass_index: i32::from(key.pass_index),
                status: CompletionStatus::Completed,
            },
            &EvalInsert {
                answer: record.answer.clone(),
                ref_answer: record.ref_answer.clone(),
                is_passed: record.is_passed,
                fail_reason: record.fail_reason.clone(),
            },
        )
        .await?;

        if let Some(checker_runtime) = checker_runtime.as_ref().filter(|_| !record.is_passed) {
            run_and_store_checker(
                db,
                PendingChecker {
                    completions_id,
                    context: record.context.clone(),
                    answer: record.answer.clone(),
                    ref_answer: record.ref_answer.clone(),
                },
                checker_runtime.as_ref(),
            )
            .await?;
        }
    }

    Ok(AttemptOutcome {
        key,
        is_passed: record.is_passed,
    })
}

pub(crate) async fn execute_pending_checks(
    pending_checks: Vec<PendingChecker>,
    db: Db,
    checker_runtime: Arc<CheckerRuntime>,
    checker_concurrency: usize,
) -> Result<(), String> {
    let mut join_set = JoinSet::new();
    let mut pending_iter = pending_checks.into_iter();

    for _ in 0..checker_concurrency {
        if let Some(pending_check) = pending_iter.next() {
            spawn_pending_checker(
                &mut join_set,
                pending_check,
                db.clone(),
                Arc::clone(&checker_runtime),
            );
        }
    }

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(())) => {
                if let Some(next_pending) = pending_iter.next() {
                    spawn_pending_checker(
                        &mut join_set,
                        next_pending,
                        db.clone(),
                        Arc::clone(&checker_runtime),
                    );
                }
            }
            Ok(Err(err)) => {
                join_set.abort_all();
                while join_set.join_next().await.is_some() {}
                return Err(err);
            }
            Err(join_err) => {
                join_set.abort_all();
                while join_set.join_next().await.is_some() {}
                if join_err.is_panic() {
                    return Err("checker task panicked".to_string());
                }
                return Err(format!("checker task join failed: {join_err}"));
            }
        }
    }

    Ok(())
}

pub(crate) fn build_attempt_keys(
    avg_k_plan: &AvgKExecutionPlan,
    max_pass_k: u8,
) -> Vec<AttemptKey> {
    let mut attempts = Vec::with_capacity(
        avg_k_plan.repeat_count * avg_k_plan.indices.len() * usize::from(max_pass_k),
    );

    for avg_repeat_index in 0..avg_k_plan.repeat_count {
        for &index in &avg_k_plan.indices {
            for pass_index in 0..max_pass_k {
                attempts.push(AttemptKey {
                    sample_index: index,
                    avg_repeat_index,
                    pass_index,
                });
            }
        }
    }

    attempts
}

fn spawn_pending_checker(
    join_set: &mut JoinSet<Result<(), String>>,
    pending_check: PendingChecker,
    db: Db,
    checker_runtime: Arc<CheckerRuntime>,
) {
    join_set.spawn(async move {
        run_and_store_checker(&db, pending_check, checker_runtime.as_ref()).await
    });
}

async fn run_and_store_checker(
    db: &Db,
    pending_check: PendingChecker,
    checker_runtime: &CheckerRuntime,
) -> Result<(), String> {
    let _permit = Arc::clone(&checker_runtime.semaphore)
        .acquire_owned()
        .await
        .map_err(|err| format!("acquire checker semaphore failed: {err}"))?;
    let checker = run_checker(
        checker_runtime.client.as_ref(),
        &checker_runtime.model_name,
        &pending_check.context,
        &pending_check.answer,
        &pending_check.ref_answer,
    )
    .await?;

    insert_checker(
        db,
        &CheckerInsert {
            completions_id: pending_check.completions_id,
            answer_correct: checker.answer_correct,
            instruction_following_error: checker.instruction_following_error,
            world_knowledge_error: checker.world_knowledge_error,
            math_error: checker.math_error,
            reasoning_logic_error: checker.reasoning_logic_error,
            thought_contains_correct_answer: checker.thought_contains_correct_answer,
            needs_human_review: checker.needs_human_review(),
            reason: checker.reason,
        },
    )
    .await
}
