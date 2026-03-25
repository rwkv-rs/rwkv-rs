use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    dtos::health::{QueuePerfSample, QueuePerfStage},
    services::current_unix_millis,
};

pub const PERF_WINDOW_MS: u64 = 60_000;

pub type QueuePerfHistory = Arc<Mutex<VecDeque<QueuePerfSample>>>;

pub fn new_perf_history() -> QueuePerfHistory {
    Arc::new(Mutex::new(VecDeque::new()))
}

pub fn record_perf_sample(
    history: &QueuePerfHistory,
    stage: QueuePerfStage,
    batch_used: usize,
    max_batch_size: usize,
    paragraph_len: usize,
    duration: Duration,
) {
    let now = current_unix_millis();
    let duration_secs = duration.as_secs_f64().max(f64::EPSILON);
    let instant_tokens_per_sec = match stage {
        QueuePerfStage::Prefill => paragraph_len as f64 / duration_secs,
        QueuePerfStage::Decode => 1.0 / duration_secs,
    };

    let mut guard = history
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.push_back(QueuePerfSample {
        ts_unix_ms: now,
        stage,
        duration_ms: duration_secs * 1000.0,
        instant_tokens_per_sec,
        batch_used,
        max_batch_size,
        batch_utilization: if max_batch_size == 0 {
            0.0
        } else {
            batch_used as f64 / max_batch_size as f64
        },
    });
    trim_perf_history(&mut guard, now);
}

pub fn snapshot_perf_history(history: &QueuePerfHistory) -> Vec<QueuePerfSample> {
    let now = current_unix_millis();
    let mut guard = history
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    trim_perf_history(&mut guard, now);
    guard.iter().cloned().collect()
}

fn trim_perf_history(history: &mut VecDeque<QueuePerfSample>, now: u64) {
    while history
        .front()
        .is_some_and(|sample| now.saturating_sub(sample.ts_unix_ms) > PERF_WINDOW_MS)
    {
        history.pop_front();
    }
}
