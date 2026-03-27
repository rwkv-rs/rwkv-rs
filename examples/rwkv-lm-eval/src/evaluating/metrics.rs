use std::collections::BTreeMap;

use rwkv_eval::datasets::BenchmarkInfo;

use super::{runtime::AttemptKey, sampling::AvgKExecutionPlan};

pub(crate) struct ComputedMetrics {
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
    let mut pass_at_k_hits = BTreeMap::<u8, usize>::new();
    let mut passed = 0usize;

    for avg_repeat_index in 0..avg_k_plan.repeat_count {
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

            if sample_attempts.iter().any(|&passed| passed) {
                passed += 1;
            }
            for &pass_k in benchmark_info.pass_ks {
                if sample_attempts
                    .iter()
                    .take(pass_k as usize)
                    .any(|&passed| passed)
                {
                    *pass_at_k_hits.entry(pass_k).or_insert(0) += 1;
                }
            }
        }
    }

    let total = avg_k_plan.repeat_count * avg_k_plan.indices.len();

    Ok(ComputedMetrics {
        pass_at_k_hits,
        passed,
        total,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use rwkv_eval::datasets::{
        Benchmark,
        BenchmarkInfo,
        BenchmarkName,
        CoTMode,
        Field,
        SamplingConfig,
    };

    use super::compute_metrics;
    use crate::evaluating::{runtime::AttemptKey, sampling::AvgKExecutionPlan};

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
