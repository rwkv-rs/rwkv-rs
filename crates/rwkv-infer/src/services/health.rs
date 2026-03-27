use std::{
    collections::{BTreeMap, VecDeque},
    sync::Mutex,
};

use all_smi::{AllSmi, device::GpuInfo};

use crate::{
    dtos::health::{GpuHealthSeries, GpuSample, GpuSampleStatus, HealthResp, QueueHealthSeries},
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
    series: BTreeMap<String, GpuSeriesState>,
}

#[derive(Clone, Debug, Default)]
struct GpuSeriesState {
    name: String,
    vendor: String,
    samples: VecDeque<GpuSample>,
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
            .map(|(device_key, series)| GpuHealthSeries {
                device_key: device_key.clone(),
                name: series.name.clone(),
                vendor: series.vendor.clone(),
                samples: series.samples.iter().cloned().collect(),
            })
            .collect()
    }
}

pub fn health(queues: &QueueMap, gpu_metrics: &GpuMetricsCache) -> HealthResp {
    let mut queue_series = Vec::new();
    for (model_name, handles) in queues {
        for handle in handles {
            queue_series.push(QueueHealthSeries {
                model_name: model_name.clone(),
                device_id: handle.device_id,
                weights_path: handle.weights_path.clone(),
                accepting: handle.is_accepting(),
                pending: handle.load_score(),
                max_batch_size: handle.max_batch_size,
                samples: handle.perf_samples(),
            });
        }
    }
    queue_series.sort_by(|left, right| {
        left.model_name
            .cmp(&right.model_name)
            .then(left.device_id.cmp(&right.device_id))
            .then(left.weights_path.cmp(&right.weights_path))
    });

    HealthResp {
        status: "ok".to_string(),
        window_seconds: HEALTH_WINDOW_SECONDS,
        server_time_unix_ms: current_unix_millis(),
        queues: queue_series,
        gpus: gpu_metrics.snapshot(),
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

        let entry = cache.series.entry(device_key).or_default();
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

fn trim_gpu_series(series: &mut BTreeMap<String, GpuSeriesState>, now: u64) {
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
