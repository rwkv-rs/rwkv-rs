use std::collections::{BTreeMap, BTreeSet};

use crate::db::{
    Db, TaskIdentity, TaskInsert, TaskLookup, TaskStatus, delete_score_by_task_id,
    find_tasks_by_identity, insert_task, list_attempt_records, update_task_status,
};
use super::{
    options::RunMode,
    runtime::{AttemptKey, PendingChecker, TaskExecutionState},
};

pub(crate) async fn prepare_task_execution(
    db: &Db,
    run_mode: RunMode,
    skip_checker: bool,
    identity: TaskIdentity,
    insert: TaskInsert,
) -> Result<TaskExecutionState, String> {
    let matches = find_tasks_by_identity(db, &identity).await?;
    match run_mode {
        RunMode::New => {
            if !matches.is_empty() {
                return Err(format!(
                    "run_mode=new refused because matching task(s) already exist: {}",
                    render_task_lookup_list(&matches)
                ));
            }

            let task_id = insert_task(db, &insert).await?;
            Ok(TaskExecutionState {
                task_id: Some(task_id),
                results: BTreeMap::new(),
                pending_checks: Vec::new(),
            })
        }
        RunMode::Resume => {
            if matches
                .iter()
                .any(|task| task.status == TaskStatus::Completed)
            {
                return Err(format!(
                    "run_mode=resume refused because a matching completed task already exists: {}",
                    render_task_lookup_list(&matches)
                ));
            }

            let resumable = matches
                .into_iter()
                .filter(|task| matches!(task.status, TaskStatus::Running | TaskStatus::Failed))
                .collect::<Vec<_>>();
            if resumable.is_empty() {
                return Err("run_mode=resume could not find a matching running/failed task".into());
            }
            if resumable.len() != 1 {
                return Err(format!(
                    "run_mode=resume is ambiguous because multiple matching running/failed tasks exist: {}",
                    render_task_lookup_list(&resumable)
                ));
            }

            let task_id = resumable[0].task_id;
            update_task_status(db, task_id, TaskStatus::Running).await?;
            delete_score_by_task_id(db, task_id).await?;
            let existing = list_attempt_records(db, task_id).await?;

            Ok(TaskExecutionState {
                task_id: Some(task_id),
                results: existing
                    .iter()
                    .map(|record| (AttemptKey::from(record.key), record.is_passed))
                    .collect(),
                pending_checks: if skip_checker {
                    Vec::new()
                } else {
                    existing
                        .into_iter()
                        .filter(|record| !record.is_passed && !record.has_checker)
                        .map(|record| PendingChecker {
                            completions_id: record.completions_id,
                            context: record.context,
                            answer: record.answer,
                            ref_answer: record.ref_answer,
                        })
                        .collect()
                },
            })
        }
        RunMode::Rerun => {
            let task_id = insert_task(db, &insert).await?;
            Ok(TaskExecutionState {
                task_id: Some(task_id),
                results: BTreeMap::new(),
                pending_checks: Vec::new(),
            })
        }
    }
}

pub(crate) fn ensure_existing_results_match_plan(
    results: &BTreeMap<AttemptKey, bool>,
    allowed: &BTreeSet<AttemptKey>,
    benchmark_name: &str,
    model_name: &str,
) {
    let extras = results
        .keys()
        .filter(|key| !allowed.contains(key))
        .copied()
        .collect::<Vec<_>>();
    assert!(
        extras.is_empty(),
        "stored attempts do not match current execution plan for benchmark `{benchmark_name}` model `{model_name}`: {extras:?}"
    );
}

fn render_task_lookup_list(tasks: &[TaskLookup]) -> String {
    tasks
        .iter()
        .map(|task| format!("task_id={} status={}", task.task_id, task.status.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}
