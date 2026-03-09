use std::path::PathBuf;
use rwkv_config::validated::eval::{FinalEvalConfig, FinalEvalConfigBuilder};
use rwkv_eval::datasets::{build_benchmark, ensure_dataset_ready, expand_benchmark_names};
use rwkv_eval::error::EvalError;
use rwkv_eval::runner::{EvalReport, ModelRunSummary};
use rwkv_eval::runtime::EvalRuntime;


pub async fn evaluating(
    eval_cfg_builder: &FinalEvalConfigBuilder,
    datasets_path: PathBuf,
) -> Result<EvalReport, EvalError> {
    eval_cfg_builder.build();

    let benchmark_names = expand_benchmark_names(&eval_cfg.benchmark_field, &eval_cfg.extra_benchmark_name)?;
    // let runtime = EvalRuntime::new(cfg, datasets_path).await?;

    let mut model_runs = Vec::new();
    for model in runtime.models() {
        println!("evaluating model: {}", model.target.display_name());
        let mut benchmark_summaries = Vec::new();

        for benchmark_name in &benchmark_names {
            println!("  benchmark: {benchmark_name}");
            let mut benchmark = build_benchmark(benchmark_name, runtime.datasets_path())?;
            ensure_dataset_ready(benchmark.as_mut())?;
            let evaluator = benchmark.get_evaluator();

            let mut results = Vec::new();
            for split in benchmark.splits() {
                for index in 0..benchmark.len(*split) {
                    results.push(
                        evaluator
                            .evaluate(&runtime, benchmark.as_ref(), model, *split, index)
                            .await?,
                    );
                }
            }

            benchmark_summaries.push(build_summary(
                benchmark_name,
                benchmark.avg_k(),
                benchmark.pass_k(),
                &results,
            ));
        }

        model_runs.push(ModelRunSummary {
            model_name: model.target.display_name(),
            benchmark_summaries,
        });
    }

    Ok(EvalReport {
        experiment_name: eval_cfg.experiment_name.clone(),
        benchmark_names,
        model_runs,
    })
}