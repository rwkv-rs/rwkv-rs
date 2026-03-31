use std::{
    collections::{BTreeMap, VecDeque},
    sync::Mutex,
};

use all_smi::{AllSmi, device::GpuInfo};

use crate::{
    dtos::health::{
        GpuHealthPanel,
        GpuHealthSeries,
        GpuHealthSummary,
        GpuSample,
        GpuSampleStatus,
        HealthResp,
        QueueHealthBinding,
        QueuePerfSample,
        QueuePerfStage,
        QueuePerfSummary,
    },
    services::{QueueMap, current_unix_millis},
};

pub const HEALTH_WINDOW_SECONDS: u64 = 60;
const HEALTH_WINDOW_MS: u64 = HEALTH_WINDOW_SECONDS * 1000;
const GPU_SAMPLE_TTL_MS: u64 = 1000;

pub struct GpuMetricsCache {
    inner: Mutex<GpuMetricsCacheInner>,
}

struct GpuMetricsCacheInner {
    smi: Option<AllSmi>,
    last_sample_unix_ms: Option<u64>,
    series: BTreeMap<u32, GpuSeriesState>,
}

#[derive(Clone, Debug, Default)]
struct GpuSeriesState {
    device_key: String,
    name: String,
    vendor: String,
    samples: VecDeque<GpuSample>,
}

#[derive(Clone, Debug)]
struct QueueHealthSnapshot {
    model_name: String,
    device_id: u32,
    weights_path: String,
    accepting: bool,
    pending: usize,
    max_batch_size: usize,
    paragraph_len: usize,
    samples: Vec<QueuePerfSample>,
}

impl Default for GpuMetricsCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMetricsCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(GpuMetricsCacheInner {
                smi: None,
                last_sample_unix_ms: None,
                series: BTreeMap::new(),
            }),
        }
    }

    pub fn snapshot(&self) -> Vec<GpuHealthSeries> {
        let now = current_unix_millis();
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        trim_gpu_series(&mut guard.series, now);

        if guard
            .last_sample_unix_ms
            .map_or(true, |last| now.saturating_sub(last) >= GPU_SAMPLE_TTL_MS)
        {
            sample_gpu_metrics(&mut guard, now);
            guard.last_sample_unix_ms = Some(now);
            trim_gpu_series(&mut guard.series, now);
        }

        guard
            .series
            .iter()
            .map(|(&device_id, series)| GpuHealthSeries {
                device_id,
                device_key: series.device_key.clone(),
                name: series.name.clone(),
                vendor: series.vendor.clone(),
                samples: series.samples.iter().cloned().collect(),
            })
            .collect()
    }
}

pub fn health(queues: &QueueMap, gpu_metrics: &GpuMetricsCache) -> HealthResp {
    let queue_snapshots = collect_queue_snapshots(queues);
    let gpu_series = gpu_metrics.snapshot();
    let server_time_unix_ms = current_unix_millis();
    build_health_response(queue_snapshots, gpu_series, server_time_unix_ms)
}

fn collect_queue_snapshots(queues: &QueueMap) -> Vec<QueueHealthSnapshot> {
    let mut queue_snapshots = Vec::new();
    for (model_name, handles) in queues {
        for handle in handles {
            queue_snapshots.push(QueueHealthSnapshot {
                model_name: model_name.clone(),
                device_id: handle.device_id,
                weights_path: handle.weights_path.clone(),
                accepting: handle.is_accepting(),
                pending: handle.load_score(),
                max_batch_size: handle.max_batch_size,
                paragraph_len: handle.paragraph_len,
                samples: handle.perf_samples(),
            });
        }
    }
    queue_snapshots
}

fn build_health_response(
    queue_snapshots: Vec<QueueHealthSnapshot>,
    gpu_series: Vec<GpuHealthSeries>,
    server_time_unix_ms: u64,
) -> HealthResp {
    let mut panels = BTreeMap::new();

    for gpu in gpu_series {
        let latest_gpu = latest_gpu_summary(&gpu.samples, server_time_unix_ms);
        panels.insert(
            gpu.device_id,
            GpuHealthPanel {
                device_id: gpu.device_id,
                device_key: gpu.device_key,
                name: gpu.name,
                vendor: gpu.vendor,
                latest_gpu,
                samples: gpu.samples,
                queues: Vec::new(),
            },
        );
    }

    for queue in queue_snapshots {
        panels
            .entry(queue.device_id)
            .or_insert_with(|| unknown_gpu_panel(queue.device_id))
            .queues
            .push(build_queue_health_binding(queue, server_time_unix_ms));
    }

    for panel in panels.values_mut() {
        panel.queues.sort_by(|left, right| {
            left.model_name
                .cmp(&right.model_name)
                .then(left.weights_path.cmp(&right.weights_path))
                .then(left.device_id.cmp(&right.device_id))
        });
    }

    HealthResp {
        status: "ok".to_string(),
        window_seconds: HEALTH_WINDOW_SECONDS,
        server_time_unix_ms,
        gpu_panels: panels.into_values().collect(),
    }
}

fn build_queue_health_binding(
    queue: QueueHealthSnapshot,
    server_time_unix_ms: u64,
) -> QueueHealthBinding {
    let latest_batch_idle_rate = queue
        .samples
        .last()
        .map(|sample| 1.0 - sample.batch_utilization);
    let latest_prefill =
        latest_queue_summary(&queue.samples, QueuePerfStage::Prefill, server_time_unix_ms);
    let latest_decode =
        latest_queue_summary(&queue.samples, QueuePerfStage::Decode, server_time_unix_ms);

    QueueHealthBinding {
        model_name: queue.model_name,
        device_id: queue.device_id,
        weights_path: queue.weights_path,
        accepting: queue.accepting,
        pending: queue.pending,
        max_batch_size: queue.max_batch_size,
        paragraph_len: queue.paragraph_len,
        latest_prefill,
        latest_decode,
        latest_batch_idle_rate,
        samples: queue.samples,
    }
}

fn latest_queue_summary(
    samples: &[QueuePerfSample],
    wanted_stage: QueuePerfStage,
    server_time_unix_ms: u64,
) -> Option<QueuePerfSummary> {
    samples
        .iter()
        .rev()
        .find(|sample| sample.stage == wanted_stage)
        .map(|sample| QueuePerfSummary {
            ts_unix_ms: sample.ts_unix_ms,
            age_ms: server_time_unix_ms.saturating_sub(sample.ts_unix_ms),
            duration_ms: sample.duration_ms,
            speed_per_sec: sample.instant_tokens_per_sec,
            instant_tokens_per_sec: sample.instant_tokens_per_sec,
            batch_used: sample.batch_used,
            max_batch_size: sample.max_batch_size,
            batch_utilization: sample.batch_utilization,
        })
}

fn latest_gpu_summary(samples: &[GpuSample], server_time_unix_ms: u64) -> Option<GpuHealthSummary> {
    samples.last().map(|sample| GpuHealthSummary {
        ts_unix_ms: sample.ts_unix_ms,
        age_ms: server_time_unix_ms.saturating_sub(sample.ts_unix_ms),
        utilization_percent: sample.utilization_percent,
        memory_used_bytes: sample.memory_used_bytes,
        memory_total_bytes: sample.memory_total_bytes,
        memory_utilization_percent: sample.memory_utilization_percent,
        status: sample.status,
    })
}

fn unknown_gpu_panel(device_id: u32) -> GpuHealthPanel {
    GpuHealthPanel {
        device_id,
        device_key: format!("device-{device_id}"),
        name: "unknown".to_string(),
        vendor: "unknown".to_string(),
        latest_gpu: None,
        samples: Vec::new(),
        queues: Vec::new(),
    }
}

fn sample_gpu_metrics(cache: &mut GpuMetricsCacheInner, now: u64) {
    if cache.smi.is_none() {
        cache.smi = AllSmi::new().ok();
    }

    let Some(smi) = cache.smi.as_ref() else {
        return;
    };

    let gpus = smi.get_gpu_info();
    if gpus.is_empty() {
        return;
    }

    for (index, info) in gpus.into_iter().enumerate() {
        let device_id = index as u32;
        let device_key = build_device_key(index, &info);
        let vendor = detect_vendor(&info);
        let memory_utilization_percent = if info.total_memory == 0 {
            None
        } else {
            Some((info.used_memory as f64 / info.total_memory as f64) * 100.0)
        };
        let status = if info.utilization == 0.0 && info.used_memory == 0 && info.total_memory == 0 {
            GpuSampleStatus::Unavailable
        } else {
            GpuSampleStatus::Ok
        };

        let entry = cache.series.entry(device_id).or_default();
        entry.device_key = device_key;
        entry.name = info.name;
        entry.vendor = vendor;
        entry.samples.push_back(GpuSample {
            ts_unix_ms: now,
            utilization_percent: Some(info.utilization),
            memory_used_bytes: Some(info.used_memory),
            memory_total_bytes: Some(info.total_memory),
            memory_utilization_percent,
            status,
        });
    }
}

fn trim_gpu_series(series: &mut BTreeMap<u32, GpuSeriesState>, now: u64) {
    series.retain(|_, state| {
        while state
            .samples
            .front()
            .is_some_and(|sample| now.saturating_sub(sample.ts_unix_ms) > HEALTH_WINDOW_MS)
        {
            state.samples.pop_front();
        }
        !state.samples.is_empty()
    });
}

fn build_device_key(index: usize, info: &GpuInfo) -> String {
    if !info.uuid.trim().is_empty() {
        info.uuid.clone()
    } else if !info.instance.trim().is_empty() {
        format!("{}:{index}", info.instance)
    } else {
        format!("{}-{index}", detect_vendor(info))
    }
}

fn detect_vendor(info: &GpuInfo) -> String {
    if let Some(lib_name) = info.detail.get("lib_name") {
        return lib_name.to_ascii_lowercase();
    }

    let lower_name = info.name.to_ascii_lowercase();
    if lower_name.contains("nvidia")
        || lower_name.contains("rtx")
        || lower_name.contains("gtx")
        || lower_name.contains("tesla")
    {
        return "nvidia".to_string();
    }
    if lower_name.contains("amd")
        || lower_name.contains("radeon")
        || lower_name.contains("instinct")
        || lower_name.contains("mi")
    {
        return "amd".to_string();
    }
    if lower_name.contains("apple")
        || lower_name.starts_with("m1")
        || lower_name.starts_with("m2")
        || lower_name.starts_with("m3")
        || lower_name.starts_with("m4")
    {
        return "apple".to_string();
    }

    info.device_type.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{QueueHealthSnapshot, build_health_response};
    use crate::dtos::health::{
        GpuHealthSeries,
        GpuSample,
        GpuSampleStatus,
        QueuePerfSample,
        QueuePerfStage,
    };

    #[test]
    fn build_health_response_binds_queues_to_gpu_panels() {
        let resp = build_health_response(
            vec![
                QueueHealthSnapshot {
                    model_name: "demo-a".to_string(),
                    device_id: 1,
                    weights_path: "/tmp/a.bpk".to_string(),
                    accepting: true,
                    pending: 2,
                    max_batch_size: 8,
                    paragraph_len: 16,
                    samples: vec![
                        queue_sample(80, QueuePerfStage::Prefill, 10.0, 1600.0, 0.75),
                        queue_sample(95, QueuePerfStage::Decode, 5.0, 200.0, 0.5),
                    ],
                },
                QueueHealthSnapshot {
                    model_name: "demo-b".to_string(),
                    device_id: 3,
                    weights_path: "/tmp/b.bpk".to_string(),
                    accepting: false,
                    pending: 0,
                    max_batch_size: 4,
                    paragraph_len: 32,
                    samples: vec![queue_sample(
                        70,
                        QueuePerfStage::Prefill,
                        20.0,
                        1600.0,
                        0.25,
                    )],
                },
            ],
            vec![
                GpuHealthSeries {
                    device_id: 2,
                    device_key: "gpu-2".to_string(),
                    name: "RTX 6000".to_string(),
                    vendor: "nvidia".to_string(),
                    samples: vec![gpu_sample(98, Some(66.0), Some(75.0))],
                },
                GpuHealthSeries {
                    device_id: 1,
                    device_key: "gpu-1".to_string(),
                    name: "RTX 4090".to_string(),
                    vendor: "nvidia".to_string(),
                    samples: vec![gpu_sample(97, Some(88.0), Some(55.0))],
                },
            ],
            100,
        );

        assert_eq!(resp.gpu_panels.len(), 3);
        assert_eq!(
            resp.gpu_panels
                .iter()
                .map(|panel| panel.device_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        let gpu_one = &resp.gpu_panels[0];
        assert_eq!(gpu_one.name, "RTX 4090");
        assert_eq!(gpu_one.latest_gpu.as_ref().map(|gpu| gpu.age_ms), Some(3));
        assert_eq!(gpu_one.queues.len(), 1);
        assert_eq!(
            gpu_one.queues[0]
                .latest_prefill
                .as_ref()
                .map(|summary| summary.speed_per_sec),
            Some(1600.0)
        );
        assert_eq!(
            gpu_one.queues[0]
                .latest_decode
                .as_ref()
                .map(|summary| summary.speed_per_sec),
            Some(200.0)
        );
        assert_eq!(gpu_one.queues[0].latest_batch_idle_rate, Some(0.5));
        assert_eq!(gpu_one.queues[0].paragraph_len, 16);

        let gpu_two = &resp.gpu_panels[1];
        assert_eq!(gpu_two.device_id, 2);
        assert!(gpu_two.queues.is_empty());

        let gpu_three = &resp.gpu_panels[2];
        assert_eq!(gpu_three.name, "unknown");
        assert!(gpu_three.latest_gpu.is_none());
        assert_eq!(gpu_three.queues.len(), 1);
    }

    fn queue_sample(
        ts_unix_ms: u64,
        stage: QueuePerfStage,
        duration_ms: f64,
        instant_tokens_per_sec: f64,
        batch_utilization: f64,
    ) -> QueuePerfSample {
        QueuePerfSample {
            ts_unix_ms,
            stage,
            duration_ms,
            instant_tokens_per_sec,
            batch_used: 1,
            max_batch_size: 4,
            batch_utilization,
        }
    }

    fn gpu_sample(
        ts_unix_ms: u64,
        utilization_percent: Option<f64>,
        memory_utilization_percent: Option<f64>,
    ) -> GpuSample {
        GpuSample {
            ts_unix_ms,
            utilization_percent,
            memory_used_bytes: Some(1024),
            memory_total_bytes: Some(2048),
            memory_utilization_percent,
            status: GpuSampleStatus::Ok,
        }
    }
}
