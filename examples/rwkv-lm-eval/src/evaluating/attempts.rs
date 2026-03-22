use std::collections::BTreeMap;
use std::sync::Arc;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use rwkv_eval::checkers::run_checker;
use rwkv_eval::datasets::{Benchmark, CoTMode};
use tokio::task::JoinSet;

use crate::db::{
    CheckerInsert, CompletionInsert, CompletionStatus, Db, EvalInsert, insert_checker,
    insert_completion_and_eval,
};

use super::client::ClientWithConfig;
use super::runtime::{AttemptKey, AttemptOutcome, CheckerRuntime, PendingChecker};
use super::sampling::AvgKExecutionPlan;

pub(crate) async fn execute_attempts(
    benchmark: Arc<dyn Benchmark>,
    target_model: Arc<ClientWithConfig>,
    judger_model_name: Option<&str>,
    judger_client: Option<Arc<Client<OpenAIConfig>>>,
    checker_runtime: Option<Arc<CheckerRuntime>>,
    cot_mode: CoTMode,
    n_shot: u8,
    pending_attempts: Vec<AttemptKey>,
    db: Option<Db>,
    task_id: Option<i32>,
    attempt_concurrency: usize,
    task_results: &mut BTreeMap<AttemptKey, bool>,
) -> Result<(), String> {
    let mut join_set = JoinSet::new();
    let mut pending_iter = pending_attempts.into_iter();

    for _ in 0..attempt_concurrency {
        if let Some(key) = pending_iter.next() {
            spawn_attempt(
                &mut join_set,
                Arc::clone(&benchmark),
                Arc::clone(&target_model),
                judger_model_name.map(ToOwned::to_owned),
                judger_client.clone(),
                checker_runtime.clone(),
                cot_mode,
                n_shot,
                db.clone(),
                task_id,
                key,
            );
        }
    }

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(outcome)) => {
                if !outcome.is_passed && db.is_none() {
                    print_failed_attempt(&outcome);
                }
                task_results.insert(outcome.key, outcome.is_passed);
                if let Some(next_key) = pending_iter.next() {
                    spawn_attempt(
                        &mut join_set,
                        Arc::clone(&benchmark),
                        Arc::clone(&target_model),
                        judger_model_name.map(ToOwned::to_owned),
                        judger_client.clone(),
                        checker_runtime.clone(),
                        cot_mode,
                        n_shot,
                        db.clone(),
                        task_id,
                        next_key,
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
                    return Err("benchmark attempt panicked".to_string());
                }
                return Err(format!("benchmark attempt join failed: {join_err}"));
            }
        }
    }

    Ok(())
}

pub(crate) async fn execute_pending_checks(
    pending_checks: Vec<PendingChecker>,
    db: Db,
    checker_runtime: Arc<CheckerRuntime>,
    attempt_concurrency: usize,
) -> Result<(), String> {
    let mut join_set = JoinSet::new();
    let mut pending_iter = pending_checks.into_iter();

    for _ in 0..attempt_concurrency {
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

fn spawn_attempt(
    join_set: &mut JoinSet<Result<AttemptOutcome, String>>,
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
) {
    join_set.spawn(async move {
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
            fail_reason: (!record.is_passed).then(|| record.fail_reason.clone()),
            context: (!record.is_passed).then(|| record.context.clone()),
        })
    });
}

fn print_failed_attempt(outcome: &AttemptOutcome) {
    println!(
        "    failed sample_index={} avg_repeat_index={} pass_index={}",
        outcome.key.sample_index, outcome.key.avg_repeat_index, outcome.key.pass_index,
    );
    println!(
        "      fail_reason:\n{}",
        indent_block(
            &truncate_for_log(outcome.fail_reason.as_deref().unwrap_or_default(), 600,),
            "        "
        )
    );
    println!(
        "      context:\n{}",
        indent_block(
            &truncate_for_log(outcome.context.as_deref().unwrap_or_default(), 2000),
            "        "
        )
    );
}

fn truncate_for_log(value: &str, max_chars: usize) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "<empty>".to_string();
    }

    let mut chars = trimmed.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}\n...<truncated>")
    } else {
        truncated
    }
}

fn indent_block(value: &str, prefix: &str) -> String {
    value
        .lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
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

fn is_content_filter_checker_error(err: &str) -> bool {
    let lowered = err.to_ascii_lowercase();
    lowered.contains("content_filter")
        || lowered.contains("content management policy")
        || lowered.contains("prompt was filtered")
        || lowered.contains("response was filtered")
}

fn build_content_filter_checker_insert(completions_id: i32, err: &str) -> CheckerInsert {
    let _ = err;
    CheckerInsert {
        completions_id,
        answer_correct: false,
        instruction_following_error: false,
        world_knowledge_error: false,
        math_error: false,
        reasoning_logic_error: false,
        thought_contains_correct_answer: false,
        needs_human_review: true,
        reason: "content_filter".to_string(),
    }
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
    let checker = match run_checker(
        checker_runtime.client.as_ref(),
        &checker_runtime.model_name,
        &pending_check.context,
        &pending_check.answer,
        &pending_check.ref_answer,
    )
    .await
    {
        Ok(checker) => checker,
        Err(err) if is_content_filter_checker_error(&err) => {
            println!(
                "checker content filter for completions_id={}: marking for human review",
                pending_check.completions_id
            );
            return insert_checker(
                db,
                &build_content_filter_checker_insert(pending_check.completions_id, &err),
            )
            .await;
        }
        Err(err) => return Err(err),
    };

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

#[cfg(test)]
mod tests {
    use super::{build_content_filter_checker_insert, is_content_filter_checker_error};

    #[test]
    fn detects_azure_content_filter_errors() {
        let err = "checker request failed: new_api_error: The response was filtered due to the prompt triggering Azure OpenAI's content management policy. (code: content_filter)";
        assert!(is_content_filter_checker_error(err));
    }

    #[test]
    fn content_filter_checker_insert_requires_human_review() {
        let insert = build_content_filter_checker_insert(42, "content_filter");
        assert_eq!(insert.completions_id, 42);
        assert!(insert.needs_human_review);
        assert_eq!(insert.reason, "content_filter");
        assert!(!insert.answer_correct);
        assert!(!insert.thought_contains_correct_answer);
    }
}
