use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthResp {
    pub status: String,
    pub window_seconds: u64,
    pub server_time_unix_ms: u64,
    pub queues: Vec<QueueHealthSeries>,
    pub gpus: Vec<GpuHealthSeries>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueHealthSeries {
    pub model_name: String,
    pub device_id: u32,
    pub weights_path: String,
    pub accepting: bool,
    pub pending: usize,
    pub max_batch_size: usize,
    pub samples: Vec<QueuePerfSample>,
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
    pub device_key: String,
    pub name: String,
    pub vendor: String,
    pub samples: Vec<GpuSample>,
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
        GpuHealthSeries,
        GpuSample,
        GpuSampleStatus,
        HealthResp,
        QueueHealthSeries,
        QueuePerfSample,
        QueuePerfStage,
    };

    #[test]
    fn serializes_health_response() {
        let resp = HealthResp {
            status: "ok".to_string(),
            window_seconds: 60,
            server_time_unix_ms: 1_746_000_000_000,
            queues: vec![QueueHealthSeries {
                model_name: "rwkv".to_string(),
                device_id: 0,
                weights_path: "/tmp/model.bpk".to_string(),
                accepting: true,
                pending: 1,
                max_batch_size: 4,
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
            gpus: vec![GpuHealthSeries {
                device_key: "gpu-0".to_string(),
                name: "RTX".to_string(),
                vendor: "nvidia".to_string(),
                samples: vec![GpuSample {
                    ts_unix_ms: 1_746_000_000_000,
                    utilization_percent: Some(42.0),
                    memory_used_bytes: Some(1024),
                    memory_total_bytes: Some(2048),
                    memory_utilization_percent: Some(50.0),
                    status: GpuSampleStatus::Ok,
                }],
            }],
        };
        let json = sonic_rs::to_string(&resp).expect("serialize health response");
        assert!(json.contains(r#""window_seconds":60"#));
        assert!(json.contains(r#""stage":"prefill""#));
        assert!(json.contains(r#""device_key":"gpu-0""#));
    }
}
