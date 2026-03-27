use std::collections::BTreeMap;

use rwkv_eval::datasets::{BenchmarkInfo, CoTMode};
use sonic_rs::json;

use super::{paths::cot_mode_name, sampling::AvgKExecutionPlan};

pub(crate) fn build_task_sampling_config_json(
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

pub(crate) fn build_metrics_json(
    benchmark_info: &BenchmarkInfo,
    avg_k_plan: &AvgKExecutionPlan,
    max_pass_k: u8,
    pass_at_k: &BTreeMap<u8, f64>,
    passed: usize,
    total: usize,
) -> String {
    let pass_at_k = benchmark_info
        .pass_ks
        .iter()
        .map(|&pass_k| {
            (
                format!("pass@{pass_k}"),
                pass_at_k.get(&pass_k).copied().unwrap_or(0.0),
            )
        })
        .collect::<BTreeMap<_, _>>();

    sonic_rs::to_string(&json!({
        "passed": passed,
        "total": total,
        "sample_size": avg_k_plan.indices.len(),
        "avg_repeat_count": avg_k_plan.repeat_count,
        "max_pass_k": max_pass_k,
        "pass_at_k": pass_at_k,
    }))
    .unwrap()
}
