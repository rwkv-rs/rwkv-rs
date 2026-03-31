use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthResp {
    pub status: String,
    pub window_seconds: u64,
    pub server_time_unix_ms: u64,
    pub gpu_panels: Vec<GpuHealthPanel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuHealthPanel {
    pub device_id: u32,
    pub device_key: String,
    pub name: String,
    pub vendor: String,
    pub latest_gpu: Option<GpuHealthSummary>,
    pub samples: Vec<GpuSample>,
    pub queues: Vec<QueueHealthBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueHealthBinding {
    pub model_name: String,
    pub device_id: u32,
    pub weights_path: String,
    pub accepting: bool,
    pub pending: usize,
    pub max_batch_size: usize,
    pub paragraph_len: usize,
    pub latest_prefill: Option<QueuePerfSummary>,
    pub latest_decode: Option<QueuePerfSummary>,
    pub latest_batch_idle_rate: Option<f64>,
    pub samples: Vec<QueuePerfSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueuePerfSummary {
    pub ts_unix_ms: u64,
    pub age_ms: u64,
    pub duration_ms: f64,
    pub speed_per_sec: f64,
    pub instant_tokens_per_sec: f64,
    pub batch_used: usize,
    pub max_batch_size: usize,
    pub batch_utilization: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueuePerfSample {
    pub ts_unix_ms: u64,
    pub stage: QueuePerfStage,
    pub duration_ms: f64,
    pub instant_tokens_per_sec: f64,
    pub batch_used: usize,
    pub max_batch_size: usize,
    pub batch_utilization: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QueuePerfStage {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuHealthSeries {
    pub device_id: u32,
    pub device_key: String,
    pub name: String,
    pub vendor: String,
    pub samples: Vec<GpuSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuHealthSummary {
    pub ts_unix_ms: u64,
    pub age_ms: u64,
    pub utilization_percent: Option<f64>,
    pub memory_used_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
    pub memory_utilization_percent: Option<f64>,
    pub status: GpuSampleStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuSample {
    pub ts_unix_ms: u64,
    pub utilization_percent: Option<f64>,
    pub memory_used_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
    pub memory_utilization_percent: Option<f64>,
    pub status: GpuSampleStatus,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuSampleStatus {
    Ok,
    Unavailable,
}

#[cfg(test)]
mod tests {
    use super::{
        GpuHealthPanel,
        GpuHealthSummary,
        GpuSample,
        GpuSampleStatus,
        HealthResp,
        QueueHealthBinding,
        QueuePerfSample,
        QueuePerfStage,
        QueuePerfSummary,
    };

    #[test]
    fn serializes_health_response() {
        let resp = HealthResp {
            status: "ok".to_string(),
            window_seconds: 60,
            server_time_unix_ms: 1_746_000_000_000,
            gpu_panels: vec![GpuHealthPanel {
                device_id: 0,
                device_key: "gpu-0".to_string(),
                name: "RTX".to_string(),
                vendor: "nvidia".to_string(),
                latest_gpu: Some(GpuHealthSummary {
                    ts_unix_ms: 1_746_000_000_000,
                    age_ms: 0,
                    utilization_percent: Some(42.0),
                    memory_used_bytes: Some(1024),
                    memory_total_bytes: Some(2048),
                    memory_utilization_percent: Some(50.0),
                    status: GpuSampleStatus::Ok,
                }),
                samples: vec![GpuSample {
                    ts_unix_ms: 1_746_000_000_000,
                    utilization_percent: Some(42.0),
                    memory_used_bytes: Some(1024),
                    memory_total_bytes: Some(2048),
                    memory_utilization_percent: Some(50.0),
                    status: GpuSampleStatus::Ok,
                }],
                queues: vec![QueueHealthBinding {
                    model_name: "rwkv".to_string(),
                    device_id: 0,
                    weights_path: "/tmp/model.bpk".to_string(),
                    accepting: true,
                    pending: 1,
                    max_batch_size: 4,
                    paragraph_len: 16,
                    latest_prefill: Some(QueuePerfSummary {
                        ts_unix_ms: 1_746_000_000_000,
                        age_ms: 0,
                        duration_ms: 12.5,
                        speed_per_sec: 20.0,
                        instant_tokens_per_sec: 20.0,
                        batch_used: 2,
                        max_batch_size: 4,
                        batch_utilization: 0.5,
                    }),
                    latest_decode: None,
                    latest_batch_idle_rate: Some(0.5),
                    samples: vec![QueuePerfSample {
                        ts_unix_ms: 1_746_000_000_000,
                        stage: QueuePerfStage::Prefill,
                        duration_ms: 12.5,
                        instant_tokens_per_sec: 20.0,
                        batch_used: 2,
                        max_batch_size: 4,
                        batch_utilization: 0.5,
                    }],
                }],
            }],
        };
        let json = sonic_rs::to_string(&resp).expect("serialize health response");
        assert!(json.contains(r#""window_seconds":60"#));
        assert!(json.contains(r#""gpu_panels":["#));
        assert!(json.contains(r#""latest_prefill":{"#));
        assert!(json.contains(r#""stage":"prefill""#));
        assert!(json.contains(r#""device_key":"gpu-0""#));
    }
}
