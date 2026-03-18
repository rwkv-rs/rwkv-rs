use crate::checker::run_checker;
use crate::db::{
    BenchmarkInsert, CheckerInsert, CompletionInsert, CompletionKey, CompletionStatus, Db,
    EvalInsert, ModelInsert, ScoreInsert, TaskIdentity, TaskInsert, TaskLookup, TaskStatus,
    connect, delete_score_by_task_id, find_tasks_by_identity, insert_checker,
    insert_completion_and_eval, insert_score, insert_task, list_attempt_records,
    update_task_status, upsert_benchmark, upsert_model,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use rwkv_config::raw::eval::{IntApiConfig, SpaceDbConfig};
use rwkv_config::validated::eval::{EVAL_CFG, FinalEvalConfig, FinalEvalConfigBuilder};
use rwkv_eval::datasets::maths::set_llm_judger_semaphore;
use rwkv_eval::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, CoTMode, Field, get_benchmarks_with_field,
};
use rwkv_eval::evaluators::coding::ensure_microsandbox_available;
use sonic_rs::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio::time::{Duration, sleep};

const AVG_K_SAMPLE_BASE_SEED: u64 = 0xA11CE5EED5EED123;
const DEFAULT_ATTEMPT_CONCURRENCY: usize = 8;
const DEFAULT_JUDGER_CONCURRENCY: usize = 8;
const DEFAULT_CHECKER_CONCURRENCY: usize = 8;
const DEFAULT_DB_POOL_MAX_CONNECTIONS: u32 = 32;
const EVALUATOR_NAME: &str = "rwkv-lm-eval";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RunMode {
    #[default]
    New,
    Resume,
    Rerun,
}

impl RunMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "new" => Ok(Self::New),
            "resume" => Ok(Self::Resume),
            "rerun" => Ok(Self::Rerun),
            other => Err(format!(
                "unsupported run mode `{other}`; expected one of: new, resume, rerun"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Resume => "resume",
            Self::Rerun => "rerun",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EvaluatingOptions {
    pub run_mode: RunMode,
    pub attempt_concurrency: usize,
    pub judger_concurrency: usize,
    pub checker_concurrency: usize,
    pub db_pool_max_connections: u32,
}

impl Default for EvaluatingOptions {
    fn default() -> Self {
        Self {
            run_mode: RunMode::New,
            attempt_concurrency: DEFAULT_ATTEMPT_CONCURRENCY,
            judger_concurrency: DEFAULT_JUDGER_CONCURRENCY,
            checker_concurrency: DEFAULT_CHECKER_CONCURRENCY,
            db_pool_max_connections: DEFAULT_DB_POOL_MAX_CONNECTIONS,
        }
    }
}

struct ClientWithConfig {
    api_cfg: IntApiConfig,
    client: Client<OpenAIConfig>,
}

struct AvgKExecutionPlan {
    repeat_count: usize,
    indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AttemptKey {
    sample_index: usize,
    avg_repeat_index: usize,
    pass_index: u8,
}

impl From<CompletionKey> for AttemptKey {
    fn from(value: CompletionKey) -> Self {
        Self {
            sample_index: value.sample_index as usize,
            avg_repeat_index: value.avg_repeat_index as usize,
            pass_index: value.pass_index as u8,
        }
    }
}

struct AttemptOutcome {
    key: AttemptKey,
    is_passed: bool,
}

struct PendingChecker {
    completions_id: i32,
    context: String,
    answer: String,
    ref_answer: String,
}

struct CheckerRuntime {
    model_name: String,
    client: Arc<Client<OpenAIConfig>>,
    semaphore: Arc<Semaphore>,
}

struct TaskExecutionState {
    task_id: Option<i32>,
    results: BTreeMap<AttemptKey, bool>,
    pending_checks: Vec<PendingChecker>,
}

pub async fn evaluating(
    eval_cfg_builder: FinalEvalConfigBuilder,
    datasets_path: PathBuf,
    config_path: PathBuf,
    logs_path: PathBuf,
) {
    eval_cfg_builder.build();

    let eval_cfg = EVAL_CFG.get().unwrap();
    let options = build_evaluating_options(eval_cfg);
    let experiment_name = eval_cfg.experiment_name.clone();
    let experiment_desc = eval_cfg.experiment_desc.clone();
    let git_hash = eval_cfg.git_hash.clone();
    let db = connect_db_if_configured(
        eval_cfg.upload_to_space,
        eval_cfg.space_db.as_ref(),
        options.db_pool_max_connections,
    )
    .await;

    if db.is_none() && options.run_mode == RunMode::Resume {
        panic!("run_mode=resume requires database persistence");
    }

    let target_models = collect_models();
    assert!(
        !target_models.is_empty(),
        "no target model matched model_arch_versions/model_data_versions/model_num_params"
    );

    let clients_with_cfg = target_models
        .into_iter()
        .map(|api_cfg| {
            Arc::new(ClientWithConfig {
                client: build_client(&api_cfg.base_url, &api_cfg.api_key),
                api_cfg,
            })
        })
        .collect::<Vec<_>>();
    let llm_judger_cfg = eval_cfg.llm_judger.clone();
    let llm_checker_cfg = eval_cfg.llm_checker.clone();
    let judger_semaphore = Arc::new(Semaphore::new(options.judger_concurrency));
    set_llm_judger_semaphore(Arc::clone(&judger_semaphore));
    let checker_runtime = db.as_ref().map(|_| {
        Arc::new(CheckerRuntime {
            model_name: llm_checker_cfg.model.clone(),
            client: Arc::new(build_client(
                &llm_checker_cfg.base_url,
                &llm_checker_cfg.api_key,
            )),
            semaphore: Arc::new(Semaphore::new(options.checker_concurrency)),
        })
    });
    let llm_judger_client = Arc::new(build_client(
        &llm_judger_cfg.base_url,
        &llm_judger_cfg.api_key,
    ));

    println!("experiment: {experiment_name}");
    println!("run mode: {}", options.run_mode.as_str());
    println!("attempt concurrency: {}", options.attempt_concurrency);
    println!("judger concurrency: {}", options.judger_concurrency);
    println!("checker concurrency: {}", options.checker_concurrency);
    println!(
        "db pool max connections: {}",
        options.db_pool_max_connections
    );
    println!("target models: {}", clients_with_cfg.len());

    for target_model in &clients_with_cfg {
        check_client(
            &target_model.api_cfg.base_url,
            &target_model.api_cfg.api_key,
            &target_model.api_cfg.model,
        )
        .await;
    }
    check_client(
        &llm_judger_cfg.base_url,
        &llm_judger_cfg.api_key,
        &llm_judger_cfg.model,
    )
    .await;
    if let Some(checker_runtime) = checker_runtime.as_ref() {
        let _ = checker_runtime;
        check_client(
            &llm_checker_cfg.base_url,
            &llm_checker_cfg.api_key,
            &llm_checker_cfg.model,
        )
        .await;
    }

    let mut model_ids = BTreeMap::new();
    if let Some(db) = db.as_ref() {
        for target_model in &clients_with_cfg {
            let model_id = upsert_model(
                db,
                &ModelInsert {
                    data_version: target_model.api_cfg.model_data_version.clone(),
                    arch_version: target_model.api_cfg.model_arch_version.clone(),
                    num_params: target_model.api_cfg.model_num_params.clone(),
                    model_name: target_model.api_cfg.model.clone(),
                },
            )
            .await
            .unwrap_or_else(|err| panic!("failed to persist model metadata: {err}"));
            model_ids.insert(model_cache_key(&target_model.api_cfg), model_id);
        }
    }

    let benchmark_infos = collect_benchmarks();
    assert!(!benchmark_infos.is_empty(), "no benchmark selected");
    ensure_microsandbox_for_coding_benchmarks(&benchmark_infos).await;

    for benchmark_info in benchmark_infos {
        validate_benchmark_info(benchmark_info);

        println!("prepare benchmark: {}", benchmark_info.name.0);
        let mut benchmark_box = (benchmark_info.create)(datasets_path.clone());
        prepare_benchmark(benchmark_info, benchmark_box.as_mut()).await;
        let benchmark: Arc<dyn Benchmark> = Arc::from(benchmark_box);

        let benchmark_id = if let Some(db) = db.as_ref() {
            let benchmark_id = upsert_benchmark(
                db,
                &BenchmarkInsert {
                    benchmark_name: benchmark_info.name.0.to_string(),
                    benchmark_split: "test".to_string(),
                    url: None,
                    status: "Completed".to_string(),
                    num_samples: benchmark.len() as i32,
                },
            )
            .await
            .unwrap_or_else(|err| panic!("failed to persist benchmark metadata: {err}"));
            Some(benchmark_id)
        } else {
            None
        };

        let max_pass_k = benchmark_info
            .pass_ks
            .iter()
            .copied()
            .max()
            .unwrap_or_else(|| panic!("benchmark `{}` has empty pass_ks", benchmark_info.name.0));
        let judger_client = benchmark_info
            .with_llm_judger
            .then(|| Arc::clone(&llm_judger_client));
        let judger_model_name = benchmark_info
            .with_llm_judger
            .then_some(llm_judger_cfg.model.clone());
        let checker_model_name = checker_runtime
            .as_ref()
            .map(|runtime| runtime.model_name.clone());

        for target_model in &clients_with_cfg {
            println!(
                "run benchmark={} model={}",
                benchmark_info.name.0, target_model.api_cfg.model,
            );

            let model_id = model_ids
                .get(&model_cache_key(&target_model.api_cfg))
                .copied();

            for &cot_mode in benchmark_info.cot_mode {
                for &n_shot in benchmark_info.n_shots {
                    for &avg_k in benchmark_info.avg_ks {
                        let avg_k_plan = build_avg_k_execution_plan(
                            benchmark_info.name.0,
                            benchmark.len(),
                            avg_k,
                        );
                        let sampling_config_json = build_task_sampling_config_json(
                            benchmark_info,
                            cot_mode,
                            n_shot,
                            avg_k,
                            judger_model_name.as_deref(),
                            checker_model_name.as_deref(),
                        );
                        let log_path = build_task_log_path(
                            &logs_path,
                            &experiment_name,
                            benchmark_info.name.0,
                            &target_model.api_cfg.model,
                            cot_mode,
                            n_shot,
                            avg_k,
                        );
                        let task_state = if let Some(db) = db.as_ref() {
                            let model_id = model_id.unwrap_or_else(|| {
                                panic!(
                                    "missing cached model_id for `{}`",
                                    target_model.api_cfg.model
                                )
                            });
                            let benchmark_id = benchmark_id.unwrap_or_else(|| {
                                panic!(
                                    "missing cached benchmark_id for `{}`",
                                    benchmark_info.name.0
                                )
                            });
                            prepare_task_execution(
                                db,
                                options.run_mode,
                                TaskIdentity {
                                    config_path: Some(config_path.display().to_string()),
                                    evaluator: EVALUATOR_NAME.to_string(),
                                    git_hash: git_hash.clone(),
                                    model_id,
                                    benchmark_id,
                                    sampling_config_json: sampling_config_json.clone(),
                                },
                                TaskInsert {
                                    config_path: Some(config_path.display().to_string()),
                                    evaluator: EVALUATOR_NAME.to_string(),
                                    is_param_search: false,
                                    is_tmp: false,
                                    status: TaskStatus::Running,
                                    git_hash: git_hash.clone(),
                                    model_id,
                                    benchmark_id,
                                    desc: Some(experiment_desc.clone()),
                                    sampling_config_json: sampling_config_json.clone(),
                                    log_path: log_path.clone(),
                                },
                            )
                            .await
                            .unwrap_or_else(|err| {
                                panic!(
                                    "failed to prepare task for benchmark `{}` model `{}`: {err}",
                                    benchmark_info.name.0, target_model.api_cfg.model
                                )
                            })
                        } else {
                            TaskExecutionState {
                                task_id: None,
                                results: BTreeMap::new(),
                                pending_checks: Vec::new(),
                            }
                        };

                        let all_attempts = build_attempt_keys(&avg_k_plan, max_pass_k);
                        let all_attempt_set = all_attempts.iter().copied().collect::<BTreeSet<_>>();
                        ensure_existing_results_match_plan(
                            &task_state.results,
                            &all_attempt_set,
                            benchmark_info.name.0,
                            &target_model.api_cfg.model,
                        );
                        let pending_attempts = all_attempts
                            .into_iter()
                            .filter(|key| !task_state.results.contains_key(key))
                            .collect::<Vec<_>>();

                        let mut task_results = task_state.results;
                        let task_id = task_state.task_id;
                        let pending_checks = task_state.pending_checks;

                        if !pending_attempts.is_empty() {
                            if let Err(err) = execute_attempts(
                                Arc::clone(&benchmark),
                                Arc::clone(target_model),
                                judger_model_name.as_deref(),
                                judger_client.clone(),
                                checker_runtime.clone(),
                                cot_mode,
                                n_shot,
                                pending_attempts,
                                db.clone(),
                                task_id,
                                options.attempt_concurrency.max(1),
                                &mut task_results,
                            )
                            .await
                            {
                                fail_task(db.clone(), task_id, err).await;
                            }
                        }

                        if !pending_checks.is_empty() {
                            let db = db.clone().unwrap_or_else(|| {
                                panic!("pending checker work requires database persistence")
                            });
                            if let Err(err) = execute_pending_checks(
                                pending_checks,
                                db.clone(),
                                checker_runtime.clone().unwrap_or_else(|| {
                                    panic!("pending checker work requires checker runtime")
                                }),
                                options.attempt_concurrency.max(1),
                            )
                            .await
                            {
                                fail_task(Some(db), task_id, err).await;
                            }
                        }

                        let metrics = match compute_metrics(
                            benchmark_info,
                            &avg_k_plan,
                            max_pass_k,
                            &task_results,
                        ) {
                            Ok(metrics) => metrics,
                            Err(err) => fail_task(db.clone(), task_id, err).await,
                        };

                        println!(
                            "  cot_mode={:?} n_shot={} avg_k={} sample_size={} repeats={} pass_k_max={} passed={}/{}",
                            cot_mode,
                            n_shot,
                            avg_k,
                            avg_k_plan.indices.len(),
                            avg_k_plan.repeat_count,
                            max_pass_k,
                            metrics.passed,
                            metrics.total,
                        );

                        if let (Some(db), Some(task_id)) = (db.as_ref(), task_id) {
                            if let Err(err) = insert_score(
                                db,
                                &ScoreInsert {
                                    task_id,
                                    cot_mode: cot_mode_name(cot_mode).to_string(),
                                    metrics_json: build_metrics_json(
                                        benchmark_info,
                                        &avg_k_plan,
                                        max_pass_k,
                                        &metrics.raw_success_counts,
                                        &metrics.pass_at_k_hits,
                                        metrics.passed,
                                        metrics.total,
                                    ),
                                },
                            )
                            .await
                            {
                                fail_task(Some(db.clone()), Some(task_id), err).await;
                            }

                            if let Err(err) =
                                update_task_status(db, task_id, TaskStatus::Completed).await
                            {
                                fail_task(Some(db.clone()), Some(task_id), err).await;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn build_evaluating_options(eval_cfg: &FinalEvalConfig) -> EvaluatingOptions {
    let run_mode = RunMode::parse(&eval_cfg.run_mode)
        .unwrap_or_else(|err| panic!("invalid `run_mode` in eval config: {err}"));
    assert!(
        eval_cfg.attempt_concurrency > 0,
        "`attempt_concurrency` must be > 0"
    );
    assert!(
        eval_cfg.judger_concurrency > 0,
        "`judger_concurrency` must be > 0"
    );
    assert!(
        eval_cfg.checker_concurrency > 0,
        "`checker_concurrency` must be > 0"
    );
    assert!(
        eval_cfg.db_pool_max_connections > 0,
        "`db_pool_max_connections` must be > 0"
    );

    EvaluatingOptions {
        run_mode,
        attempt_concurrency: eval_cfg.attempt_concurrency,
        judger_concurrency: eval_cfg.judger_concurrency,
        checker_concurrency: eval_cfg.checker_concurrency,
        db_pool_max_connections: eval_cfg.db_pool_max_connections,
    }
}

fn validate_benchmark_info(benchmark_info: &BenchmarkInfo) {
    assert!(
        !benchmark_info.avg_ks.is_empty(),
        "benchmark `{}` has empty avg_ks",
        benchmark_info.name.0,
    );
    assert!(
        !benchmark_info.pass_ks.is_empty(),
        "benchmark `{}` has empty pass_ks",
        benchmark_info.name.0,
    );
}

async fn prepare_task_execution(
    db: &Db,
    run_mode: RunMode,
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
                pending_checks: existing
                    .into_iter()
                    .filter(|record| !record.is_passed && !record.has_checker)
                    .map(|record| PendingChecker {
                        completions_id: record.completions_id,
                        context: record.context,
                        answer: record.answer,
                        ref_answer: record.ref_answer,
                    })
                    .collect(),
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

fn render_task_lookup_list(tasks: &[TaskLookup]) -> String {
    tasks
        .iter()
        .map(|task| format!("task_id={} status={}", task.task_id, task.status.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn ensure_existing_results_match_plan(
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

async fn execute_attempts(
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

            if !record.is_passed {
                let checker_runtime = checker_runtime
                    .as_ref()
                    .ok_or_else(|| "failed attempt requires checker runtime".to_string())?;
                if let Err(err) = run_and_store_checker(
                    db,
                    PendingChecker {
                        completions_id,
                        context: record.context.clone(),
                        answer: record.answer.clone(),
                        ref_answer: record.ref_answer.clone(),
                    },
                    checker_runtime.as_ref(),
                )
                .await
                {
                    eprintln!(
                        "checker failed for task_id={} sample_index={} avg_repeat_index={} pass_index={}: {}",
                        task_id, key.sample_index, key.avg_repeat_index, key.pass_index, err
                    );
                }
            }
        }

        Ok(AttemptOutcome {
            key,
            is_passed: record.is_passed,
        })
    });
}

async fn execute_pending_checks(
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

fn spawn_pending_checker(
    join_set: &mut JoinSet<Result<(), String>>,
    pending_check: PendingChecker,
    db: Db,
    checker_runtime: Arc<CheckerRuntime>,
) {
    join_set.spawn(async move {
        if let Err(err) = run_and_store_checker(&db, pending_check, checker_runtime.as_ref()).await
        {
            eprintln!("checker failed while backfilling pending rows: {err}");
        }
        Ok(())
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
        Err(err) => {
            eprintln!(
                "checker request degraded to default row with reason=0 for completions_id={}: {err}",
                pending_check.completions_id
            );
            crate::checker::CheckerOutput {
                answer_correct: false,
                instruction_following_error: false,
                world_knowledge_error: false,
                math_error: false,
                reasoning_logic_error: false,
                thought_contains_correct_answer: false,
                reason: "0".to_string(),
            }
        }
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

struct ComputedMetrics {
    raw_success_counts: Vec<Vec<u8>>,
    pass_at_k_hits: BTreeMap<u8, usize>,
    passed: usize,
    total: usize,
}

fn compute_metrics(
    benchmark_info: &BenchmarkInfo,
    avg_k_plan: &AvgKExecutionPlan,
    max_pass_k: u8,
    results: &BTreeMap<AttemptKey, bool>,
) -> Result<ComputedMetrics, String> {
    let mut raw_success_counts = Vec::with_capacity(avg_k_plan.repeat_count);
    let mut pass_at_k_hits = BTreeMap::<u8, usize>::new();

    for avg_repeat_index in 0..avg_k_plan.repeat_count {
        let mut success_counts = Vec::with_capacity(avg_k_plan.indices.len());

        for &index in &avg_k_plan.indices {
            let mut sample_attempts = Vec::with_capacity(max_pass_k as usize);
            for pass_index in 0..max_pass_k {
                let key = AttemptKey {
                    sample_index: index,
                    avg_repeat_index,
                    pass_index,
                };
                let is_passed = results.get(&key).copied().ok_or_else(|| {
                    format!(
                        "missing attempt result for sample_index={} avg_repeat_index={} pass_index={}",
                        key.sample_index, key.avg_repeat_index, key.pass_index
                    )
                })?;
                sample_attempts.push(is_passed);
            }

            let success_count = sample_attempts.iter().filter(|&&passed| passed).count() as u8;
            for &pass_k in benchmark_info.pass_ks {
                if sample_attempts
                    .iter()
                    .take(pass_k as usize)
                    .any(|&passed| passed)
                {
                    *pass_at_k_hits.entry(pass_k).or_insert(0) += 1;
                }
            }
            success_counts.push(success_count);
        }

        raw_success_counts.push(success_counts);
    }

    let passed = raw_success_counts
        .iter()
        .flatten()
        .filter(|&&success_count| success_count > 0)
        .count();
    let total = avg_k_plan.repeat_count * avg_k_plan.indices.len();

    Ok(ComputedMetrics {
        raw_success_counts,
        pass_at_k_hits,
        passed,
        total,
    })
}

fn build_attempt_keys(avg_k_plan: &AvgKExecutionPlan, max_pass_k: u8) -> Vec<AttemptKey> {
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

fn collect_models() -> Vec<IntApiConfig> {
    let mut target_models = Vec::new();
    for model_arch_version in &EVAL_CFG.get().unwrap().model_arch_versions {
        for model_data_version in &EVAL_CFG.get().unwrap().model_data_versions {
            for model_num_param in &EVAL_CFG.get().unwrap().model_num_params {
                target_models.extend(
                    EVAL_CFG
                        .get()
                        .unwrap()
                        .models
                        .iter()
                        .filter(|model| {
                            model.model_arch_version == *model_arch_version
                                && model.model_data_version == *model_data_version
                                && model.model_num_params == *model_num_param
                        })
                        .cloned(),
                );
            }
        }
    }

    target_models
}

fn collect_benchmarks() -> Vec<&'static BenchmarkInfo> {
    let mut benchmark_infos = BTreeMap::new();

    for benchmark_field in &EVAL_CFG.get().unwrap().benchmark_field {
        for benchmark_info in get_benchmarks_with_field(parse_field(benchmark_field)) {
            benchmark_infos.insert(benchmark_info.name.0, *benchmark_info);
        }
    }

    for benchmark_name in &EVAL_CFG.get().unwrap().extra_benchmark_name {
        let benchmark_info = ALL_BENCHMARKS
            .iter()
            .find(|benchmark_info| benchmark_info.name.0 == benchmark_name)
            .unwrap_or_else(|| panic!("unknown benchmark `{benchmark_name}`"));
        benchmark_infos.insert(benchmark_info.name.0, benchmark_info);
    }

    benchmark_infos.into_values().collect()
}

async fn ensure_microsandbox_for_coding_benchmarks(benchmark_infos: &[&'static BenchmarkInfo]) {
    if !benchmark_infos
        .iter()
        .any(|benchmark_info| benchmark_info.field == Field::Coding)
    {
        return;
    }

    if ensure_microsandbox_available().await.is_ok() {
        return;
    }

    start_microsandbox_service().unwrap_or_else(|err| {
        panic!(
            "coding benchmark requires microsandbox, but automatic startup failed: {err}. \
please check microsandbox installation and ensure `msb server start --dev` or `microsandbox-server` is available."
        )
    });

    let mut last_err = String::new();
    for _ in 0..20 {
        match ensure_microsandbox_available().await {
            Ok(()) => return,
            Err(err) => last_err = err,
        }
        sleep(Duration::from_millis(500)).await;
    }

    panic!(
        "coding benchmark requires microsandbox, but the service is still unavailable after automatic startup: {}. \
please check microsandbox installation and server startup logs.",
        last_err
    );
}

fn start_microsandbox_service() -> Result<(), String> {
    let mut errors = Vec::new();

    for (program, args) in [
        ("msb", &["server", "start", "--dev"][..]),
        ("microsandbox-server", &[][..]),
    ] {
        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        match command.spawn() {
            Ok(_) => return Ok(()),
            Err(err) => {
                let rendered_args = if args.is_empty() {
                    String::new()
                } else {
                    format!(" {}", args.join(" "))
                };
                errors.push(format!("`{program}{rendered_args}`: {err}"));
            }
        }
    }

    Err(errors.join("; "))
}

fn parse_field(field_name: &str) -> Field {
    match field_name.trim() {
        "Knowledge" => Field::Knowledge,
        "Math" | "Maths" => Field::Maths,
        "Coding" => Field::Coding,
        "Instruction Following" | "InstructionFollowing" => Field::InstructionFollowing,
        "Function Call" | "FunctionCalling" => Field::FunctionCalling,
        _ => panic!("unknown benchmark field `{field_name}`"),
    }
}

fn build_client(base_url: &str, api_key: &str) -> Client<OpenAIConfig> {
    let config = OpenAIConfig::new()
        .with_api_key(api_key.to_string())
        .with_api_base(norm_api_url(base_url));

    Client::with_config(config)
}

fn norm_api_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    assert!(!base_url.is_empty(), "base_url cannot be empty");

    let base_url = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("http://{base_url}")
    };
    let base_url = base_url.trim_end_matches('/').to_string();

    if base_url.ends_with("/v1") {
        base_url
    } else {
        format!("{base_url}/v1")
    }
}

async fn check_client(base_url: &str, api_key: &str, model_name: &str) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|err| panic!("failed to build preflight http client: {err}"));
    let base_url = norm_api_url(base_url);

    let completion_body = sonic_rs::to_string(&json!({
        "model": model_name,
        "prompt": "ping",
        "stop": ["\n"],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    }))
    .unwrap();
    if send_preflight_request(
        &client,
        &format!("{base_url}/completions"),
        api_key,
        completion_body,
    )
    .await
    .is_ok()
    {
        return;
    }

    let chat_body = sonic_rs::to_string(&json!({
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    }))
    .unwrap();
    send_preflight_request(
        &client,
        &format!("{base_url}/chat/completions"),
        api_key,
        chat_body,
    )
    .await
    .unwrap_or_else(|error| panic!("client `{model_name}` is unavailable: {error}"));
}

async fn send_preflight_request(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: String,
) -> Result<(), String> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if !api_key.is_empty() {
        let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|err| format!("invalid authorization header: {err}"))?;
        headers.insert(AUTHORIZATION, value);
    }

    let response = client
        .post(url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|err| format!("request to `{url}` failed: {err}"))?;
    if response.status().is_success() {
        return Ok(());
    }

    let status = response.status();
    let text = response
        .text()
        .await
        .unwrap_or_else(|_| "<failed to read response body>".to_string());
    Err(format!("request to `{url}` returned {status}: {text}"))
}

fn build_avg_k_execution_plan(
    benchmark_name: &str,
    benchmark_len: usize,
    avg_k: f32,
) -> AvgKExecutionPlan {
    assert!(
        avg_k.is_finite() && avg_k > 0.0,
        "benchmark `{benchmark_name}` has invalid avg_k={avg_k}; avg_k must be finite and > 0"
    );
    assert!(
        benchmark_len > 0,
        "benchmark `{benchmark_name}` has no samples to evaluate"
    );

    if avg_k < 1.0 {
        let sample_size = compute_ratio_sample_size(benchmark_len, avg_k);
        let seed = AVG_K_SAMPLE_BASE_SEED
            ^ fnv1a_hash64(benchmark_name.as_bytes())
            ^ u64::from(avg_k.to_bits());
        AvgKExecutionPlan {
            repeat_count: 1,
            indices: deterministic_sample_indices(benchmark_len, sample_size, seed),
        }
    } else {
        let repeat_count = parse_avg_k_repeat_count(benchmark_name, avg_k);
        AvgKExecutionPlan {
            repeat_count,
            indices: (0..benchmark_len).collect(),
        }
    }
}

fn parse_avg_k_repeat_count(benchmark_name: &str, avg_k: f32) -> usize {
    let rounded = avg_k.round();
    assert!(
        (avg_k - rounded).abs() <= f32::EPSILON,
        "benchmark `{benchmark_name}` has invalid avg_k={avg_k}; avg_k >= 1 must be an integer repeat count"
    );

    rounded as usize
}

fn compute_ratio_sample_size(total_len: usize, ratio: f32) -> usize {
    (((total_len as f64) * f64::from(ratio)).round() as usize).clamp(1, total_len)
}

fn deterministic_sample_indices(total_len: usize, sample_size: usize, seed: u64) -> Vec<usize> {
    assert!(
        sample_size <= total_len,
        "sample_size={sample_size} exceeds total_len={total_len}"
    );

    let mut indices = (0..total_len).collect::<Vec<_>>();
    let mut rng = SplitMix64::new(seed);

    for start in 0..sample_size {
        let remaining = total_len - start;
        let offset = (rng.next_u64() % remaining as u64) as usize;
        indices.swap(start, start + offset);
    }

    indices.truncate(sample_size);
    indices.sort_unstable();
    indices
}

fn fnv1a_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3_u64);
    }
    hash
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

async fn prepare_benchmark(benchmark_info: &BenchmarkInfo, benchmark: &mut dyn Benchmark) {
    let load_invalid = benchmark.load();
    let check_invalid = if load_invalid {
        true
    } else {
        benchmark.check().await
    };

    if load_invalid || check_invalid {
        benchmark.download().await;

        let load_invalid = benchmark.load();
        let check_invalid = if load_invalid {
            true
        } else {
            benchmark.check().await
        };

        assert!(
            !load_invalid && !check_invalid,
            "benchmark `{}` is still invalid after download",
            benchmark_info.name.0,
        );
    }
}

async fn connect_db_if_configured(
    upload_to_space: bool,
    cfg: Option<&SpaceDbConfig>,
    max_connections: u32,
) -> Option<Db> {
    if !upload_to_space {
        println!("database persistence: disabled");
        return None;
    }

    let Some(cfg) = cfg else {
        panic!("upload_to_space=true requires [space_db] config");
    };
    if !is_space_db_configured(cfg) {
        panic!("upload_to_space=true requires a complete [space_db] config");
    }

    let db = connect(cfg, max_connections)
        .await
        .unwrap_or_else(|err| panic!("failed to connect to postgres: {err}"));
    println!("database persistence: enabled (pool max connections = {max_connections})");
    Some(db)
}

fn is_space_db_configured(cfg: &SpaceDbConfig) -> bool {
    !cfg.host.trim().is_empty()
        && !cfg.username.trim().is_empty()
        && !cfg.password.trim().is_empty()
        && !cfg.port.trim().is_empty()
        && !cfg.database_name.trim().is_empty()
}

fn model_cache_key(api_cfg: &IntApiConfig) -> String {
    format!(
        "{}|{}|{}|{}",
        api_cfg.model_arch_version,
        api_cfg.model_data_version,
        api_cfg.model_num_params,
        api_cfg.model
    )
}

fn build_task_sampling_config_json(
    benchmark_info: &BenchmarkInfo,
    cot_mode: CoTMode,
    n_shot: u8,
    avg_k: f32,
    judger_model_name: Option<&str>,
    checker_model_name: Option<&str>,
) -> String {
    sonic_rs::to_string(&json!({
        "cot_mode": cot_mode_name(cot_mode),
        "n_shot": n_shot,
        "avg_k": avg_k,
        "pass_ks": benchmark_info.pass_ks,
        "sampling_config": {
            "temperature": benchmark_info.sampling_config.temperature,
            "top_k": benchmark_info.sampling_config.top_k,
            "top_p": benchmark_info.sampling_config.top_p,
            "presence_penalty": benchmark_info.sampling_config.presence_penalty,
            "repetition_penalty": benchmark_info.sampling_config.repetition_penalty,
            "penalty_decay": benchmark_info.sampling_config.penalty_decay,
        },
        "judger_model_name": judger_model_name,
        "checker_model_name": checker_model_name,
    }))
    .unwrap()
}

fn build_metrics_json(
    benchmark_info: &BenchmarkInfo,
    avg_k_plan: &AvgKExecutionPlan,
    max_pass_k: u8,
    raw_success_counts: &[Vec<u8>],
    pass_at_k_hits: &BTreeMap<u8, usize>,
    passed: usize,
    total: usize,
) -> String {
    let pass_at_k = benchmark_info
        .pass_ks
        .iter()
        .map(|&pass_k| {
            (
                format!("pass@{pass_k}"),
                pass_at_k_hits
                    .get(&pass_k)
                    .copied()
                    .map(|hits| hits as f64 / total as f64)
                    .unwrap_or(0.0),
            )
        })
        .collect::<BTreeMap<_, _>>();

    sonic_rs::to_string(&json!({
        "passed": passed,
        "total": total,
        "sample_size": avg_k_plan.indices.len(),
        "avg_repeat_count": avg_k_plan.repeat_count,
        "max_pass_k": max_pass_k,
        "raw_success_counts": raw_success_counts,
        "pass_at_k": pass_at_k,
    }))
    .unwrap()
}

fn build_task_log_path(
    logs_root: &Path,
    experiment_name: &str,
    benchmark_name: &str,
    model_name: &str,
    cot_mode: CoTMode,
    n_shot: u8,
    avg_k: f32,
) -> String {
    logs_root
        .join(sanitize_path_component(experiment_name))
        .join(sanitize_path_component(benchmark_name))
        .join(format!(
            "{}_{}_nshot{}_avgk{}.log",
            sanitize_path_component(model_name),
            cot_mode_name(cot_mode).to_ascii_lowercase(),
            n_shot,
            sanitize_path_component(&format!("{avg_k}")),
        ))
        .display()
        .to_string()
}

fn cot_mode_name(cot_mode: CoTMode) -> &'static str {
    match cot_mode {
        CoTMode::NoCoT => "NoCoT",
        CoTMode::FakeCoT => "FakeCoT",
        CoTMode::CoT => "CoT",
    }
}

fn sanitize_path_component(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            rendered.push(ch);
        } else {
            rendered.push('_');
        }
    }
    rendered.trim_matches('_').to_string()
}

async fn fail_task(db: Option<Db>, task_id: Option<i32>, err: String) -> ! {
    if let (Some(db), Some(task_id)) = (db.as_ref(), task_id) {
        let _ = update_task_status(db, task_id, TaskStatus::Failed).await;
    }
    panic!("{err}");
}
