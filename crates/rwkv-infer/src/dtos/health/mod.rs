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

