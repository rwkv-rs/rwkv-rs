pub(crate) struct AvgKExecutionPlan {
    pub repeat_count: usize,
    pub indices: Vec<usize>,
}

const AVG_K_SAMPLE_BASE_SEED: u64 = 0xA11CE5EED5EED123;

pub(crate) fn build_avg_k_execution_plan(
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

