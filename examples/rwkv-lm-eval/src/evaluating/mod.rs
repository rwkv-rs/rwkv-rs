mod attempts;
mod benchmark;
mod client;
mod db;
mod metrics;
mod models;
mod options;
mod paths;
mod persistence_json;
mod runtime;
mod sampling;
mod task_persistence;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use rwkv_config::validated::eval::{EVAL_CFG, FinalEvalConfigBuilder};
use rwkv_eval::datasets::Benchmark;
use rwkv_eval::datasets::maths::set_llm_judger_semaphore;
use tokio::sync::Semaphore;

use crate::db::{
    BenchmarkInsert, Db, ModelInsert, ScoreInsert, TaskInsert, TaskStatus, insert_score,
    update_task_status, upsert_benchmark, upsert_model,
};

use self::attempts::{build_attempt_keys, execute_attempts, execute_pending_checks};
use self::benchmark::{
    collect_benchmarks, ensure_microsandbox_for_coding_benchmarks, prepare_benchmark,
    validate_benchmark_info,
};
use self::client::{ClientWithConfig, build_client, check_client};
use self::db::connect_db_if_configured;
use self::metrics::compute_metrics;
use self::models::{collect_models, model_cache_key};
use self::options::{RunMode, build_evaluating_options};
use self::paths::{build_task_log_path, cot_mode_name};
use self::persistence_json::{build_metrics_json, build_task_sampling_config_json};
use self::runtime::{CheckerRuntime, TaskExecutionState};
use self::sampling::build_avg_k_execution_plan;
use self::task_persistence::{
    ensure_existing_results_match_plan, fail_task, prepare_task_execution,
};

const EVALUATOR_NAME: &str = "rwkv-lm-eval";

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
        check_client(&target_model.client, &target_model.api_cfg.model).await;
    }
    check_client(&llm_judger_client, &llm_judger_cfg.model).await;
    if let Some(checker_runtime) = checker_runtime.as_ref() {
        check_client(checker_runtime.client.as_ref(), &llm_checker_cfg.model).await;
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
            run_target_model(
                benchmark_info,
                Arc::clone(&benchmark),
                benchmark_id,
                max_pass_k,
                judger_model_name.as_deref(),
                judger_client.clone(),
                checker_model_name.as_deref(),
                checker_runtime.clone(),
                target_model,
                &model_ids,
                &db,
                &options,
                &config_path,
                &logs_path,
                &experiment_name,
                &experiment_desc,
                &git_hash,
            )
            .await;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_target_model(
    benchmark_info: &'static rwkv_eval::datasets::BenchmarkInfo,
    benchmark: Arc<dyn Benchmark>,
    benchmark_id: Option<i32>,
    max_pass_k: u8,
    judger_model_name: Option<&str>,
    judger_client: Option<Arc<async_openai::Client<async_openai::config::OpenAIConfig>>>,
    checker_model_name: Option<&str>,
    checker_runtime: Option<Arc<CheckerRuntime>>,
    target_model: &Arc<ClientWithConfig>,
    model_ids: &BTreeMap<String, i32>,
    db: &Option<Db>,
    options: &options::EvaluatingOptions,
    config_path: &std::path::Path,
    logs_path: &std::path::Path,
    experiment_name: &str,
    experiment_desc: &str,
    git_hash: &str,
) {
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
                let avg_k_plan =
                    build_avg_k_execution_plan(benchmark_info.name.0, benchmark.len(), avg_k);
                let sampling_config_json = build_task_sampling_config_json(
                    benchmark_info,
                    cot_mode,
                    n_shot,
                    avg_k,
                    judger_model_name,
                    checker_model_name,
                );
                let log_path = build_task_log_path(
                    logs_path,
                    experiment_name,
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
                        crate::db::TaskIdentity {
                            config_path: Some(config_path.display().to_string()),
                            evaluator: EVALUATOR_NAME.to_string(),
                            git_hash: git_hash.to_string(),
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
                            git_hash: git_hash.to_string(),
                            model_id,
                            benchmark_id,
                            desc: Some(experiment_desc.to_string()),
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
                        judger_model_name,
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

                let metrics =
                    match compute_metrics(benchmark_info, &avg_k_plan, max_pass_k, &task_results) {
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

                    if let Err(err) = update_task_status(db, task_id, TaskStatus::Completed).await {
                        fail_task(Some(db.clone()), Some(task_id), err).await;
                    }
                }
            }
        }
    }
}
