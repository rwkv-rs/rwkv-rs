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

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::{
        cores::datasets::{
            Benchmark,
            BenchmarkInfo,
            BenchmarkName,
            CoTMode,
            Field,
            SamplingConfig,
        },
        services::runner::{runtime::AttemptKey, sampling::AvgKExecutionPlan},
    };
    use super::{compute_metrics, estimate_pass_at_k};

    #[test]
    fn computes_pass_at_k_with_combination_estimator() {
        let benchmark_info = BenchmarkInfo {
            name: BenchmarkName("demo"),
            field: Field::Maths,
            display_name: "Demo",
            cot_mode: &[CoTMode::NoCoT],
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
            pass_ks: &[1, 2],
            with_llm_judger: false,
            create: dummy_create,
        };
        let plan = AvgKExecutionPlan {
            repeat_count: 1,
            indices: vec![0, 1],
        };
        let results = BTreeMap::from([
            (
                AttemptKey {
                    sample_index: 0,
                    avg_repeat_index: 0,
                    pass_index: 0,
                },
                false,
            ),
            (
                AttemptKey {
                    sample_index: 0,
                    avg_repeat_index: 0,
                    pass_index: 1,
                },
                true,
            ),
            (
                AttemptKey {
                    sample_index: 1,
                    avg_repeat_index: 0,
                    pass_index: 0,
                },
                true,
            ),
            (
                AttemptKey {
                    sample_index: 1,
                    avg_repeat_index: 0,
                    pass_index: 1,
                },
                false,
            ),
        ]);

        let metrics = compute_metrics(&benchmark_info, &plan, 2, &results).unwrap();
        assert_eq!(metrics.passed, 2);
        assert_eq!(metrics.total, 2);
        assert_eq!(metrics.pass_at_k.get(&2), Some(&1.0));
        assert!((metrics.pass_at_k.get(&1).copied().unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn estimates_pass_at_k_from_success_count() {
        assert!((estimate_pass_at_k(3, 1, 1).unwrap() - (1.0 / 3.0)).abs() < 1e-12);
        assert!((estimate_pass_at_k(3, 1, 2).unwrap() - (2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(estimate_pass_at_k(3, 0, 2).unwrap(), 0.0);
        assert_eq!(estimate_pass_at_k(3, 2, 2).unwrap(), 1.0);
    }

    fn dummy_create(_: PathBuf) -> Box<dyn Benchmark> {
        panic!("unused in test")
    }
}
