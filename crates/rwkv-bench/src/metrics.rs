use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub request_id: usize,
    pub success: bool,
    pub error: Option<String>,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub ttft_s: Option<f64>,
    pub itl_s: Vec<f64>,
    pub tpot_s: Option<f64>,
    pub e2el_s: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PercentileSummary {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub completed: usize,
    pub failed: usize,
    pub duration_s: f64,
    pub request_throughput: f64,
    pub output_token_throughput: f64,
    pub total_token_throughput: f64,

    pub mean_ttft_ms: f64,
    pub mean_itl_ms: f64,
    pub mean_tpot_ms: f64,
    pub mean_e2el_ms: f64,

    pub ttft_ms: PercentileSummary,
    pub itl_ms: PercentileSummary,
    pub tpot_ms: PercentileSummary,
    pub e2el_ms: PercentileSummary,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn percentile(sorted_values: &[f64], pct: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let n = sorted_values.len();
    let rank = (pct / 100.0) * (n.saturating_sub(1) as f64);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        return sorted_values[lower];
    }

    let weight = rank - lower as f64;
    sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
}

fn summarize_percentiles_ms(values_s: &[f64]) -> PercentileSummary {
    if values_s.is_empty() {
        return PercentileSummary::default();
    }

    let mut v = values_s.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));

    PercentileSummary {
        p50: percentile(&v, 50.0) * 1000.0,
        p90: percentile(&v, 90.0) * 1000.0,
        p95: percentile(&v, 95.0) * 1000.0,
        p99: percentile(&v, 99.0) * 1000.0,
    }
}

pub fn aggregate_metrics(requests: &[RequestMetrics], duration_s: f64) -> AggregateMetrics {
    let completed = requests.iter().filter(|r| r.success).count();
    let failed = requests.len().saturating_sub(completed);
    let safe_duration = duration_s.max(1e-9);

    let output_tokens = requests
        .iter()
        .filter(|r| r.success)
        .map(|r| r.output_tokens)
        .sum::<usize>();
    let input_tokens = requests
        .iter()
        .filter(|r| r.success)
        .map(|r| r.prompt_tokens)
        .sum::<usize>();

    let ttft_values = requests
        .iter()
        .filter_map(|r| r.ttft_s)
        .filter(|v| *v > 0.0)
        .collect::<Vec<_>>();
    let itl_values = requests
        .iter()
        .flat_map(|r| r.itl_s.iter().copied())
        .filter(|v| *v > 0.0)
        .collect::<Vec<_>>();
    let tpot_values = requests
        .iter()
        .filter_map(|r| r.tpot_s)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let e2el_values = requests
        .iter()
        .map(|r| r.e2el_s)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();

    AggregateMetrics {
        completed,
        failed,
        duration_s,
        request_throughput: completed as f64 / safe_duration,
        output_token_throughput: output_tokens as f64 / safe_duration,
        total_token_throughput: (output_tokens + input_tokens) as f64 / safe_duration,
        mean_ttft_ms: mean(&ttft_values) * 1000.0,
        mean_itl_ms: mean(&itl_values) * 1000.0,
        mean_tpot_ms: mean(&tpot_values) * 1000.0,
        mean_e2el_ms: mean(&e2el_values) * 1000.0,
        ttft_ms: summarize_percentiles_ms(&ttft_values),
        itl_ms: summarize_percentiles_ms(&itl_values),
        tpot_ms: summarize_percentiles_ms(&tpot_values),
        e2el_ms: summarize_percentiles_ms(&e2el_values),
    }
}
