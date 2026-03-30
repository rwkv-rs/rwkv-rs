use std::{
    cmp::Ordering,
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use async_openai::{Client, config::OpenAIConfig};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{
    cores::datasets::{Benchmark, BenchmarkInfo, CoTMode},
    db::{Db, ScoreInsert, TaskStatus, insert_score, update_task_status},
    services::admin::{DesiredState, EvalRuntimeControl, ObservedStatus},
};
use super::{
    attempts::{execute_attempt, execute_pending_checks},
    client::ClientWithConfig,
    metrics::compute_metrics,
    paths::cot_mode_name,
    persistence_json::build_metrics_json,
    runtime::{AttemptKey, CheckerRuntime, PendingChecker},
    sampling::AvgKExecutionPlan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TaskRunId(pub usize);

pub(crate) struct ModelRuntime {
    pub semaphore: Arc<Semaphore>,
}

pub(crate) struct TaskRunState {
    pub benchmark_info: &'static BenchmarkInfo,
    pub benchmark: Arc<dyn Benchmark>,
    pub max_pass_k: u8,
    pub judger_model_name: Option<String>,
    pub judger_client: Option<Arc<Client<OpenAIConfig>>>,
    pub checker_runtime: Option<Arc<CheckerRuntime>>,
    pub target_model: Arc<ClientWithConfig>,
    pub model_key: String,
    pub avg_k_plan: AvgKExecutionPlan,
    pub cot_mode: CoTMode,
    pub n_shot: u8,
    pub avg_k: f32,
    pub task_id: Option<i32>,
    pub task_results: BTreeMap<AttemptKey, bool>,
    pub pending_attempts: VecDeque<AttemptKey>,
    pub inflight_attempts: usize,
    pub pending_checks: Vec<PendingChecker>,
    pub checker_running: bool,
    pub completed: bool,
    pub failed_error: Option<String>,
}

impl TaskRunState {
    fn remaining_attempts(&self) -> usize {
        self.pending_attempts.len() + self.inflight_attempts
    }

    fn is_terminal(&self) -> bool {
        self.completed || self.failed_error.is_some()
    }

    fn label(&self) -> String {
        format!(
            "benchmark={} model={} cot_mode={:?} n_shot={} avg_k={}",
            self.benchmark_info.name.0,
            self.target_model.api_cfg.model,
            self.cot_mode,
            self.n_shot,
            self.avg_k
        )
    }
}

struct AttemptCompletion {
    task_run_id: TaskRunId,
    key: AttemptKey,
    is_passed: bool,
}

struct CheckerCompletion {
    task_run_id: TaskRunId,
}

struct TaskRunError {
    task_run_id: TaskRunId,
    err: String,
}

pub(crate) async fn run_scheduler(
    task_runs: Vec<TaskRunState>,
    model_runtimes: BTreeMap<String, ModelRuntime>,
    db: Option<Db>,
    checker_concurrency: usize,
    runtime_control: Option<EvalRuntimeControl>,
) {
    let model_keys = model_runtimes.keys().cloned().collect::<Vec<_>>();
    let mut task_runs = task_runs
        .into_iter()
        .enumerate()
        .map(|(index, task_run)| (TaskRunId(index), task_run))
        .collect::<BTreeMap<_, _>>();
    let mut task_run_ids_by_model = BTreeMap::<String, Vec<TaskRunId>>::new();
    for (&task_run_id, task_run) in &task_runs {
        task_run_ids_by_model
            .entry(task_run.model_key.clone())
            .or_default()
            .push(task_run_id);
    }

    let mut attempt_runs = JoinSet::<Result<AttemptCompletion, TaskRunError>>::new();
    let mut checker_runs = JoinSet::<Result<CheckerCompletion, TaskRunError>>::new();
    let mut inflight_attempt_count = 0usize;
    let mut inflight_checker_count = 0usize;

    if let Some(control) = runtime_control.as_ref() {
        control
            .write_status(ObservedStatus::Running, None)
            .unwrap_or_else(|err| panic!("failed to write running runtime status: {err}"));
    }

    drive_ready_tasks(
        &mut task_runs,
        &db,
        checker_concurrency,
        &mut checker_runs,
        &mut inflight_checker_count,
        true,
    )
    .await;

    loop {
        let desired_state = poll_desired_state(runtime_control.as_ref());
        let allow_dispatch = matches!(desired_state, DesiredState::Running);
        update_runtime_status(
            runtime_control.as_ref(),
            desired_state,
            inflight_attempt_count,
            inflight_checker_count,
        );

        loop {
            if !allow_dispatch {
                break;
            }
            let mut dispatched_in_pass = false;
            for model_key in &model_keys {
                if dispatch_one_attempt(
                    model_key,
                    &model_runtimes,
                    &task_run_ids_by_model,
                    &mut task_runs,
                    &db,
                    &mut attempt_runs,
                ) {
                    inflight_attempt_count += 1;
                    dispatched_in_pass = true;
                }
            }

            if !dispatched_in_pass {
                break;
            }
        }

        drive_ready_tasks(
            &mut task_runs,
            &db,
            checker_concurrency,
            &mut checker_runs,
            &mut inflight_checker_count,
            allow_dispatch,
        )
        .await;

        if task_runs.values().all(TaskRunState::is_terminal) {
            break;
        }

        if desired_state == DesiredState::Cancelled
            && inflight_attempt_count == 0
            && inflight_checker_count == 0
        {
            cancel_pending_tasks(&mut task_runs, &db).await;
            if let Some(control) = runtime_control.as_ref() {
                control
                    .write_status(ObservedStatus::Cancelled, None)
                    .unwrap_or_else(|err| {
                        panic!("failed to write cancelled runtime status: {err}")
                    });
            }
            break;
        }

        if inflight_attempt_count == 0 && inflight_checker_count == 0 {
            if matches!(
                desired_state,
                DesiredState::Paused | DesiredState::Cancelled
            ) {
                tokio::time::sleep(Duration::from_millis(250)).await;
                continue;
            }
            panic!("scheduler became idle while unfinished tasks remain");
        }

        tokio::select! {
            result = attempt_runs.join_next(), if inflight_attempt_count > 0 => {
                let result = result.unwrap_or_else(|| panic!("attempt join set ended unexpectedly"));
                inflight_attempt_count -= 1;
                match result {
                    Ok(Ok(completion)) => {
                        let task_run = task_runs.get_mut(&completion.task_run_id).unwrap_or_else(|| {
                            panic!("missing task state for {:?}", completion.task_run_id)
                        });
                        task_run.inflight_attempts = task_run
                            .inflight_attempts
                            .checked_sub(1)
                            .unwrap_or_else(|| panic!("attempt underflow for {}", task_run.label()));
                        if !task_run.is_terminal() {
                            task_run
                                .task_results
                                .insert(completion.key, completion.is_passed);
                        }
                    }
                    Ok(Err(task_err)) => {
                        handle_task_error(&mut task_runs, &db, task_err, true).await;
                    }
                    Err(join_err) => {
                        if join_err.is_panic() {
                            panic!("attempt task panicked");
                        }
                        panic!("attempt task join failed: {join_err}");
                    }
                }
            }
            result = checker_runs.join_next(), if inflight_checker_count > 0 => {
                let result = result.unwrap_or_else(|| panic!("checker join set ended unexpectedly"));
                inflight_checker_count -= 1;
                match result {
                    Ok(Ok(completion)) => {
                        let task_run = task_runs.get_mut(&completion.task_run_id).unwrap_or_else(|| {
                            panic!("missing task state for {:?}", completion.task_run_id)
                        });
                        task_run.checker_running = false;
                    }
                    Ok(Err(task_err)) => {
                        handle_task_error(&mut task_runs, &db, task_err, false).await;
                    }
                    Err(join_err) => {
                        if join_err.is_panic() {
                            panic!("checker task panicked");
                        }
                        panic!("checker task join failed: {join_err}");
                    }
                }
            }
        }
    }

    let failed_count = task_runs
        .values()
        .filter(|task_run| task_run.failed_error.is_some())
        .count();
    if failed_count > 0 {
        eprintln!("evaluation finished with {failed_count} failed task(s)");
    }

    if let Some(control) = runtime_control.as_ref() {
        let desired_state = control.desired_state().unwrap_or(DesiredState::Running);
        if desired_state != DesiredState::Cancelled {
            let message = (failed_count > 0)
                .then(|| format!("evaluation finished with {failed_count} failed task(s)"));
            let status = if failed_count > 0 {
                ObservedStatus::Failed
            } else {
                ObservedStatus::Completed
            };
            control
                .write_status(status, message.as_deref())
                .unwrap_or_else(|err| panic!("failed to write final runtime status: {err}"));
        }
    }
}

fn dispatch_one_attempt(
    model_key: &str,
    model_runtimes: &BTreeMap<String, ModelRuntime>,
    task_run_ids_by_model: &BTreeMap<String, Vec<TaskRunId>>,
    task_runs: &mut BTreeMap<TaskRunId, TaskRunState>,
    db: &Option<Db>,
    attempt_runs: &mut JoinSet<Result<AttemptCompletion, TaskRunError>>,
) -> bool {
    let Some(task_run_ids) = task_run_ids_by_model.get(model_key) else {
        return false;
    };
    let Some(task_run_id) = pick_next_task(task_run_ids, task_runs) else {
        return false;
    };
    let Some(permit) = model_runtimes
        .get(model_key)
        .unwrap_or_else(|| panic!("missing model runtime for `{model_key}`"))
        .semaphore
        .clone()
        .try_acquire_owned()
        .ok()
    else {
        return false;
    };

    let task_run = task_runs
        .get_mut(&task_run_id)
        .unwrap_or_else(|| panic!("missing task state for {:?}", task_run_id));
    let Some(key) = task_run.pending_attempts.pop_front() else {
        return false;
    };
    task_run.inflight_attempts += 1;

    let benchmark = Arc::clone(&task_run.benchmark);
    let target_model = Arc::clone(&task_run.target_model);
    let judger_model_name = task_run.judger_model_name.clone();
    let judger_client = task_run.judger_client.clone();
    let checker_runtime = task_run.checker_runtime.clone();
    let cot_mode = task_run.cot_mode;
    let n_shot = task_run.n_shot;
    let db = db.clone();
    let task_id = task_run.task_id;

    attempt_runs.spawn(async move {
        let _permit = permit;
        let outcome = execute_attempt(
            benchmark,
            target_model,
            judger_model_name,
            judger_client,
            checker_runtime,
            cot_mode,
            n_shot,
            db,
            task_id,
            key,
        )
        .await
        .map_err(|err| TaskRunError { task_run_id, err })?;

        Ok(AttemptCompletion {
            task_run_id,
            key: outcome.key,
            is_passed: outcome.is_passed,
        })
    });

    true
}

fn pick_next_task(
    task_run_ids: &[TaskRunId],
    task_runs: &BTreeMap<TaskRunId, TaskRunState>,
) -> Option<TaskRunId> {
    let mut best: Option<TaskRunId> = None;

    for &task_run_id in task_run_ids {
        let Some(candidate) = task_runs.get(&task_run_id) else {
            continue;
        };
        if candidate.is_terminal() || candidate.pending_attempts.is_empty() {
            continue;
        }

        match best {
            Some(best_id) => {
                let best_task = task_runs
                    .get(&best_id)
                    .unwrap_or_else(|| panic!("missing task state for {:?}", best_id));
                if compare_task_priority(task_run_id, candidate, best_id, best_task).is_lt() {
                    best = Some(task_run_id);
                }
            }
            None => best = Some(task_run_id),
        }
    }

    best
}

fn compare_task_priority(
    left_id: TaskRunId,
    left: &TaskRunState,
    right_id: TaskRunId,
    right: &TaskRunState,
) -> Ordering {
    left.remaining_attempts()
        .cmp(&right.remaining_attempts())
        .then_with(|| left.benchmark_info.name.0.cmp(right.benchmark_info.name.0))
        .then_with(|| left.cot_mode.cmp(&right.cot_mode))
        .then_with(|| left.n_shot.cmp(&right.n_shot))
        .then_with(|| left.avg_k.total_cmp(&right.avg_k))
        .then_with(|| {
            left.task_id
                .unwrap_or(i32::MAX)
                .cmp(&right.task_id.unwrap_or(i32::MAX))
        })
        .then_with(|| left_id.cmp(&right_id))
}

async fn drive_ready_tasks(
    task_runs: &mut BTreeMap<TaskRunId, TaskRunState>,
    db: &Option<Db>,
    checker_concurrency: usize,
    checker_runs: &mut JoinSet<Result<CheckerCompletion, TaskRunError>>,
    inflight_checker_count: &mut usize,
    allow_dispatch: bool,
) {
    let ready_ids = task_runs.keys().copied().collect::<Vec<_>>();

    for task_run_id in ready_ids {
        let action = {
            let task_run = task_runs
                .get_mut(&task_run_id)
                .unwrap_or_else(|| panic!("missing task state for {:?}", task_run_id));
            if task_run.is_terminal()
                || task_run.inflight_attempts > 0
                || !task_run.pending_attempts.is_empty()
            {
                ReadyAction::None
            } else if task_run.checker_running {
                ReadyAction::None
            } else if !allow_dispatch && !task_run.pending_checks.is_empty() {
                ReadyAction::None
            } else if !task_run.pending_checks.is_empty() {
                let db = db.clone().unwrap_or_else(|| {
                    panic!("pending checker work requires database persistence")
                });
                let checker_runtime = task_run
                    .checker_runtime
                    .clone()
                    .unwrap_or_else(|| panic!("pending checker work requires checker runtime"));
                let pending_checks = std::mem::take(&mut task_run.pending_checks);
                task_run.checker_running = true;
                ReadyAction::RunChecks {
                    db,
                    checker_runtime,
                    pending_checks,
                }
            } else {
                ReadyAction::Finalize
            }
        };

        match action {
            ReadyAction::None => {}
            ReadyAction::RunChecks {
                db,
                checker_runtime,
                pending_checks,
            } => {
                *inflight_checker_count += 1;
                checker_runs.spawn(async move {
                    execute_pending_checks(
                        pending_checks,
                        db,
                        checker_runtime,
                        checker_concurrency.max(1),
                    )
                    .await
                    .map_err(|err| TaskRunError { task_run_id, err })?;
                    Ok(CheckerCompletion { task_run_id })
                });
            }
            ReadyAction::Finalize => {
                let task_run = task_runs
                    .get_mut(&task_run_id)
                    .unwrap_or_else(|| panic!("missing task state for {:?}", task_run_id));
                if let Err(err) = finalize_task(task_run, db.as_ref()).await {
                    handle_task_error(task_runs, db, TaskRunError { task_run_id, err }, false)
                        .await;
                }
            }
        }
    }
}

async fn finalize_task(task_run: &mut TaskRunState, db: Option<&Db>) -> Result<(), String> {
    let metrics = compute_metrics(
        task_run.benchmark_info,
        &task_run.avg_k_plan,
        task_run.max_pass_k,
        &task_run.task_results,
    )?;

    println!(
        "  benchmark={} model={} cot_mode={:?} n_shot={} avg_k={} sample_size={} repeats={} pass_k_max={} passed={}/{}",
        task_run.benchmark_info.name.0,
        task_run.target_model.api_cfg.model,
        task_run.cot_mode,
        task_run.n_shot,
        task_run.avg_k,
        task_run.avg_k_plan.indices.len(),
        task_run.avg_k_plan.repeat_count,
        task_run.max_pass_k,
        metrics.passed,
        metrics.total,
    );

    if let (Some(db), Some(task_id)) = (db, task_run.task_id) {
        insert_score(
            db,
            &ScoreInsert {
                task_id,
                cot_mode: cot_mode_name(task_run.cot_mode).to_string(),
                metrics_json: build_metrics_json(
                    task_run.benchmark_info,
                    &task_run.avg_k_plan,
                    task_run.max_pass_k,
                    &metrics.pass_at_k,
                    metrics.passed,
                    metrics.total,
                ),
            },
        )
        .await?;

        update_task_status(db, task_id, TaskStatus::Completed).await?;
    }

    task_run.completed = true;
    Ok(())
}

async fn handle_task_error(
    task_runs: &mut BTreeMap<TaskRunId, TaskRunState>,
    db: &Option<Db>,
    task_err: TaskRunError,
    from_attempt: bool,
) {
    let Some(task_run) = task_runs.get_mut(&task_err.task_run_id) else {
        return;
    };
    if from_attempt && task_run.inflight_attempts > 0 {
        task_run.inflight_attempts -= 1;
    }
    if task_run.failed_error.is_some() {
        return;
    }

    let label = task_run.label();
    let task_id = task_run.task_id;
    task_run.failed_error = Some(task_err.err.clone());
    task_run.pending_attempts.clear();
    task_run.pending_checks.clear();
    task_run.checker_running = false;

    mark_task_failed(db.clone(), task_id).await;
    eprintln!("task failed: {label}: {}", task_err.err);
}

async fn mark_task_failed(db: Option<Db>, task_id: Option<i32>) {
    if let (Some(db), Some(task_id)) = (db.as_ref(), task_id) {
        let _ = update_task_status(db, task_id, TaskStatus::Failed).await;
    }
}

async fn cancel_pending_tasks(task_runs: &mut BTreeMap<TaskRunId, TaskRunState>, db: &Option<Db>) {
    for task_run in task_runs.values_mut() {
        if task_run.is_terminal() {
            continue;
        }
        task_run.pending_attempts.clear();
        task_run.pending_checks.clear();
        task_run.checker_running = false;
        task_run.failed_error = Some("cancelled by admin".to_string());
        mark_task_failed(db.clone(), task_run.task_id).await;
    }
}

fn poll_desired_state(runtime_control: Option<&EvalRuntimeControl>) -> DesiredState {
    runtime_control
        .and_then(|control| control.desired_state().ok())
        .unwrap_or(DesiredState::Running)
}

fn update_runtime_status(
    runtime_control: Option<&EvalRuntimeControl>,
    desired_state: DesiredState,
    inflight_attempt_count: usize,
    inflight_checker_count: usize,
) {
    let Some(control) = runtime_control else {
        return;
    };

    let status = match desired_state {
        DesiredState::Running => ObservedStatus::Running,
        DesiredState::Paused if inflight_attempt_count == 0 && inflight_checker_count == 0 => {
            ObservedStatus::Paused
        }
        DesiredState::Paused => ObservedStatus::Pausing,
        DesiredState::Cancelled if inflight_attempt_count == 0 && inflight_checker_count == 0 => {
            ObservedStatus::Cancelled
        }
        DesiredState::Cancelled => ObservedStatus::Cancelling,
    };

    let _ = control.write_status(status, None);
}

enum ReadyAction {
    None,
    RunChecks {
        db: Db,
        checker_runtime: Arc<CheckerRuntime>,
        pending_checks: Vec<PendingChecker>,
    },
    Finalize,
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use rwkv_config::raw::eval::IntApiConfig;

    use crate::cores::{
        datasets::{Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig},
        evaluation::client::build_client,
    };
    use super::*;

    #[test]
    fn pick_next_task_prefers_smaller_total_remaining_work() {
        let task_runs = BTreeMap::from([
            (
                TaskRunId(0),
                make_task_run("alpha", vec![0, 1], 1, CoTMode::NoCoT, 0, 1.0, Some(10)),
            ),
            (
                TaskRunId(1),
                make_task_run(
                    "beta",
                    vec![0, 1, 2, 3],
                    0,
                    CoTMode::NoCoT,
                    0,
                    1.0,
                    Some(11),
                ),
            ),
        ]);

        let picked = pick_next_task(&[TaskRunId(0), TaskRunId(1)], &task_runs);
        assert_eq!(picked, Some(TaskRunId(0)));
    }

    #[test]
    fn pick_next_task_uses_stable_tiebreakers() {
        let task_runs = BTreeMap::from([
            (
                TaskRunId(0),
                make_task_run("beta", vec![0], 0, CoTMode::CoT, 1, 2.0, Some(20)),
            ),
            (
                TaskRunId(1),
                make_task_run("alpha", vec![0], 0, CoTMode::NoCoT, 0, 1.0, Some(10)),
            ),
        ]);

        let picked = pick_next_task(&[TaskRunId(0), TaskRunId(1)], &task_runs);
        assert_eq!(picked, Some(TaskRunId(1)));
    }

    fn make_task_run(
        benchmark_name: &'static str,
        pending_samples: Vec<usize>,
        inflight_attempts: usize,
        cot_mode: CoTMode,
        n_shot: u8,
        avg_k: f32,
        task_id: Option<i32>,
    ) -> TaskRunState {
        TaskRunState {
            benchmark_info: Box::leak(Box::new(BenchmarkInfo {
                name: BenchmarkName(benchmark_name),
                field: Field::Knowledge,
                display_name: benchmark_name,
                cot_mode: &[CoTMode::NoCoT, CoTMode::CoT],
                sampling_config: SamplingConfig {
                    temperature: 1.0,
                    top_k: 1,
                    top_p: 1.0,
                    presence_penalty: 0.0,
                    repetition_penalty: 0.0,
                    penalty_decay: 1.0,
                },
                n_shots: &[0],
                avg_ks: &[1.0],
                pass_ks: &[1],
                with_llm_judger: false,
                create: dummy_create,
            })),
            benchmark: Arc::new(DummyBenchmark),
            max_pass_k: 1,
            judger_model_name: None,
            judger_client: None,
            checker_runtime: None,
            target_model: Arc::new(ClientWithConfig {
                api_cfg: IntApiConfig {
                    model_arch_version: "rwkv7".to_string(),
                    model_data_version: "g1".to_string(),
                    model_num_params: "1.5b".to_string(),
                    base_url: "127.0.0.1:8080".to_string(),
                    api_key: "test".to_string(),
                    model: "demo-model".to_string(),
                    max_batch_size: Some(8),
                },
                client: build_client("127.0.0.1:8080", "test"),
            }),
            model_key: "demo-model".to_string(),
            avg_k_plan: AvgKExecutionPlan {
                repeat_count: 1,
                indices: vec![0],
            },
            cot_mode,
            n_shot,
            avg_k,
            task_id,
            task_results: BTreeMap::new(),
            pending_attempts: pending_samples
                .into_iter()
                .map(|sample_index| AttemptKey {
                    sample_index,
                    avg_repeat_index: 0,
                    pass_index: 0,
                })
                .collect(),
            inflight_attempts,
            pending_checks: Vec::new(),
            checker_running: false,
            completed: false,
            failed_error: None,
        }
    }

    fn dummy_create(_: PathBuf) -> Box<dyn Benchmark> {
        Box::new(DummyBenchmark)
    }

    struct DummyBenchmark;

    #[async_trait::async_trait]
    impl Benchmark for DummyBenchmark {
        fn load(&mut self) -> bool {
            false
        }

        async fn check(&self) -> bool {
            false
        }

        async fn download(&self) {}

        fn len(&self) -> usize {
            1
        }

        fn get_expected_context(&self, _: usize, _: CoTMode, _: u8) -> String {
            String::new()
        }

        fn get_ref_answer(&self, _: usize) -> String {
            String::new()
        }

        async fn answer_and_judge(
            &self,
            _: &str,
            _: &Client<OpenAIConfig>,
            _: Option<&str>,
            _: Option<&Client<OpenAIConfig>>,
            _: CoTMode,
            _: u8,
            _: usize,
        ) -> crate::cores::datasets::Record {
            panic!("unused in scheduler unit tests")
        }
    }
}
