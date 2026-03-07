use std::path::PathBuf;

use rwkv_config::validated::eval::FinalEvalConfig;

use crate::datasets::{build_benchmark, ensure_dataset_ready, expand_benchmark_names};
use crate::error::EvalError;
use crate::evaluators::EvaluationSampleResult;
use crate::runtime::EvalRuntime;

#[derive(Clone, Debug)]
pub struct BenchmarkRunSummary {
    pub benchmark_name: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub accuracy: f64,
    pub avg_k: usize,
    pub pass_k: usize,
}

#[derive(Clone, Debug)]
pub struct ModelRunSummary {
    pub model_name: String,
    pub benchmark_summaries: Vec<BenchmarkRunSummary>,
}

#[derive(Clone, Debug)]
pub struct EvalReport {
    pub experiment_name: String,
    pub benchmark_names: Vec<String>,
    pub model_runs: Vec<ModelRunSummary>,
}



fn build_summary(
    benchmark_name: &str,
    avg_k: usize,
    pass_k: usize,
    results: &[EvaluationSampleResult],
) -> BenchmarkRunSummary {
    let passed = results.iter().filter(|result| result.is_pass).count();
    let total = results.len();
    let failed = total.saturating_sub(passed);
    let accuracy = if total == 0 {
        0.0
    } else {
        passed as f64 / total as f64
    };

    BenchmarkRunSummary {
        benchmark_name: benchmark_name.to_string(),
        total,
        passed,
        failed,
        accuracy,
        avg_k,
        pass_k,
    }
}
