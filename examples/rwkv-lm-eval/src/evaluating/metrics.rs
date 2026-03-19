use rwkv_eval::datasets::BenchmarkInfo;
use std::collections::BTreeMap;

use super::runtime::AttemptKey;
use super::sampling::AvgKExecutionPlan;

pub(crate) struct ComputedMetrics {
    pub raw_success_counts: Vec<Vec<u8>>,
    pub pass_at_k_hits: BTreeMap<u8, usize>,
    pub passed: usize,
    pub total: usize,
}

pub(crate) fn compute_metrics(
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

#[cfg(test)]
mod tests {
    use super::compute_metrics;
    use crate::evaluating::runtime::AttemptKey;
    use crate::evaluating::sampling::AvgKExecutionPlan;
    use rwkv_eval::datasets::{
        Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[test]
    fn computes_pass_at_k_counts() {
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
        assert_eq!(metrics.pass_at_k_hits.get(&1), Some(&1));
        assert_eq!(metrics.pass_at_k_hits.get(&2), Some(&2));
    }

    fn dummy_create(_: PathBuf) -> Box<dyn Benchmark> {
        panic!("unused in test")
    }
}
