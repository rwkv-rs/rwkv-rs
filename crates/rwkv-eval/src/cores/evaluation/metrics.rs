use std::collections::BTreeMap;

use crate::cores::datasets::BenchmarkInfo;
use super::{runtime::AttemptKey, sampling::AvgKExecutionPlan};

pub(crate) struct ComputedMetrics {
    pub pass_at_k: BTreeMap<u8, f64>,
    pub passed: usize,
    pub total: usize,
}

pub(crate) fn compute_metrics(
    benchmark_info: &BenchmarkInfo,
    avg_k_plan: &AvgKExecutionPlan,
    max_pass_k: u8,
    results: &BTreeMap<AttemptKey, bool>,
) -> Result<ComputedMetrics, String> {
    let mut pass_at_k_sums = BTreeMap::<u8, f64>::new();
    let mut passed = 0usize;
    let sampled_attempt_count = usize::from(max_pass_k);

    for avg_repeat_index in 0..avg_k_plan.repeat_count {
        for &index in &avg_k_plan.indices {
            let mut success_count = 0usize;
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
                if is_passed {
                    success_count += 1;
                }
            }

            if success_count > 0 {
                passed += 1;
            }
            for &pass_k in benchmark_info.pass_ks {
                let estimate = estimate_pass_at_k(sampled_attempt_count, success_count, pass_k)?;
                *pass_at_k_sums.entry(pass_k).or_insert(0.0) += estimate;
            }
        }
    }

    let total = avg_k_plan.repeat_count * avg_k_plan.indices.len();
    let pass_at_k = benchmark_info
        .pass_ks
        .iter()
        .copied()
        .map(|pass_k| {
            (
                pass_k,
                pass_at_k_sums.get(&pass_k).copied().unwrap_or(0.0) / total as f64,
            )
        })
        .collect();

    Ok(ComputedMetrics {
        pass_at_k,
        passed,
        total,
    })
}

fn estimate_pass_at_k(n: usize, c: usize, k: u8) -> Result<f64, String> {
    let k = usize::from(k);
    if k == 0 {
        return Err("pass@0 is invalid".to_string());
    }
    if k > n {
        return Err(format!("pass@{k} exceeds sampled attempt count n={n}"));
    }
    if c > n {
        return Err(format!(
            "success count c={c} exceeds sampled attempt count n={n}"
        ));
    }
    if c == 0 {
        return Ok(0.0);
    }
    if n - c < k {
        return Ok(1.0);
    }

    let mut fail_prob = 1.0;
    for offset in 0..k {
        fail_prob *= (n - c - offset) as f64 / (n - offset) as f64;
    }
    Ok(1.0 - fail_prob)
}

