use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StageBreakdownMs {
    pub validate_ms: Option<f64>,
    pub tokenize_ms: Option<f64>,
    pub queue_wait_ms: Option<f64>,
    pub schedule_wait_ms: Option<f64>,
    pub prefill_first_ms: Option<f64>,
    pub first_emit_ms: Option<f64>,
    pub prefill_total_ms: Option<f64>,
    pub decode_total_ms: Option<f64>,
    pub request_total_ms: Option<f64>,
}

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
    pub stage_ms: StageBreakdownMs,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PercentileSummary {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StageSummaryMs {
    pub mean_validate_ms: f64,
    pub mean_tokenize_ms: f64,
    pub mean_queue_wait_ms: f64,
    pub mean_schedule_wait_ms: f64,
    pub mean_prefill_first_ms: f64,
    pub mean_first_emit_ms: f64,
    pub mean_prefill_total_ms: f64,
    pub mean_decode_total_ms: f64,
    pub mean_request_total_ms: f64,

    pub validate_ms: PercentileSummary,
    pub tokenize_ms: PercentileSummary,
    pub queue_wait_ms: PercentileSummary,
    pub schedule_wait_ms: PercentileSummary,
    pub prefill_first_ms: PercentileSummary,
    pub first_emit_ms: PercentileSummary,
    pub prefill_total_ms: PercentileSummary,
    pub decode_total_ms: PercentileSummary,
    pub request_total_ms: PercentileSummary,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ErrorBuckets {
    pub bad_request: usize,
    pub timeout: usize,
    pub cancel: usize,
    pub backend_error: usize,
    pub other: usize,
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

    pub stage_ms: StageSummaryMs,
    pub error_rate: f64,
    pub error_buckets: ErrorBuckets,
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

fn summarize_percentiles(values: &[f64]) -> PercentileSummary {
    if values.is_empty() {
        return PercentileSummary::default();
    }

    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));

    PercentileSummary {
        p50: percentile(&v, 50.0),
        p90: percentile(&v, 90.0),
        p95: percentile(&v, 95.0),
        p99: percentile(&v, 99.0),
    }
}

fn classify_error(error: &str) -> &'static str {
    let msg = error.to_ascii_lowercase();
    if msg.contains("http 400")
        || msg.contains("invalid_request_error")
        || msg.contains("bad request")
    {
        "bad_request"
    } else if msg.contains("timeout") || msg.contains("deadline") {
        "timeout"
    } else if msg.contains("cancel") || msg.contains("canceled") {
        "cancel"
    } else if msg.contains("http 5")
        || msg.contains("internal_error")
        || msg.contains("decode failed")
    {
        "backend_error"
    } else {
        "other"
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

    let ttft_values_s = requests
        .iter()
        .filter_map(|r| r.ttft_s)
        .filter(|v| *v > 0.0)
        .collect::<Vec<_>>();
    let itl_values_s = requests
        .iter()
        .flat_map(|r| r.itl_s.iter().copied())
        .filter(|v| *v > 0.0)
        .collect::<Vec<_>>();
    let tpot_values_s = requests
        .iter()
        .filter_map(|r| r.tpot_s)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let e2el_values_s = requests
        .iter()
        .map(|r| r.e2el_s)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();

    let validate_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.validate_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let tokenize_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.tokenize_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let queue_wait_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.queue_wait_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let schedule_wait_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.schedule_wait_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let prefill_first_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.prefill_first_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let first_emit_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.first_emit_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let prefill_total_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.prefill_total_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let decode_total_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.decode_total_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();
    let request_total_values_ms = requests
        .iter()
        .filter_map(|r| r.stage_ms.request_total_ms)
        .filter(|v| *v >= 0.0)
        .collect::<Vec<_>>();

    let mut error_buckets = ErrorBuckets::default();
    for error in requests.iter().filter_map(|r| r.error.as_deref()) {
        match classify_error(error) {
            "bad_request" => error_buckets.bad_request += 1,
            "timeout" => error_buckets.timeout += 1,
            "cancel" => error_buckets.cancel += 1,
            "backend_error" => error_buckets.backend_error += 1,
            _ => error_buckets.other += 1,
        }
    }

    AggregateMetrics {
        completed,
        failed,
        duration_s,
        request_throughput: completed as f64 / safe_duration,
        output_token_throughput: output_tokens as f64 / safe_duration,
        total_token_throughput: (output_tokens + input_tokens) as f64 / safe_duration,
        mean_ttft_ms: mean(&ttft_values_s) * 1000.0,
        mean_itl_ms: mean(&itl_values_s) * 1000.0,
        mean_tpot_ms: mean(&tpot_values_s) * 1000.0,
        mean_e2el_ms: mean(&e2el_values_s) * 1000.0,
        ttft_ms: summarize_percentiles(
            &ttft_values_s.iter().map(|v| v * 1000.0).collect::<Vec<_>>(),
        ),
        itl_ms: summarize_percentiles(&itl_values_s.iter().map(|v| v * 1000.0).collect::<Vec<_>>()),
        tpot_ms: summarize_percentiles(
            &tpot_values_s.iter().map(|v| v * 1000.0).collect::<Vec<_>>(),
        ),
        e2el_ms: summarize_percentiles(
            &e2el_values_s.iter().map(|v| v * 1000.0).collect::<Vec<_>>(),
        ),
        stage_ms: StageSummaryMs {
            mean_validate_ms: mean(&validate_values_ms),
            mean_tokenize_ms: mean(&tokenize_values_ms),
            mean_queue_wait_ms: mean(&queue_wait_values_ms),
            mean_schedule_wait_ms: mean(&schedule_wait_values_ms),
            mean_prefill_first_ms: mean(&prefill_first_values_ms),
            mean_first_emit_ms: mean(&first_emit_values_ms),
            mean_prefill_total_ms: mean(&prefill_total_values_ms),
            mean_decode_total_ms: mean(&decode_total_values_ms),
            mean_request_total_ms: mean(&request_total_values_ms),
            validate_ms: summarize_percentiles(&validate_values_ms),
            tokenize_ms: summarize_percentiles(&tokenize_values_ms),
            queue_wait_ms: summarize_percentiles(&queue_wait_values_ms),
            schedule_wait_ms: summarize_percentiles(&schedule_wait_values_ms),
            prefill_first_ms: summarize_percentiles(&prefill_first_values_ms),
            first_emit_ms: summarize_percentiles(&first_emit_values_ms),
            prefill_total_ms: summarize_percentiles(&prefill_total_values_ms),
            decode_total_ms: summarize_percentiles(&decode_total_values_ms),
            request_total_ms: summarize_percentiles(&request_total_values_ms),
        },
        error_rate: if requests.is_empty() {
            0.0
        } else {
            failed as f64 / requests.len() as f64
        },
        error_buckets,
    }
}
