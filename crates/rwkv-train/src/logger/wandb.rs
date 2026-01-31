use std::{collections::HashMap, sync::{Arc, OnceLock}};
use burn::train::{
    logger::{AsyncLogger, Logger, MetricLogger},
    metric::{MetricDefinition, MetricId, NumericEntry, SerializedEntry},
    metric::store::{EpochSummary, MetricsUpdate, Split},
};
use log::warn;
use tokio::{runtime::Runtime, sync::Mutex};
use wandb::{BackendOptions, LogData, Run, RunInfo, WandB};

/// Configuration for initializing a WandB logger.
#[derive(Debug, Clone)]
pub struct WandbLoggerConfig {
    pub api_key: String,
    pub entity: Option<String>,
    pub project: String,
    pub run_name: Option<String>,
}

static WANDB_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn wandb_runtime() -> &'static Runtime {
    WANDB_RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime should be created"))
}

impl WandbLoggerConfig {
    pub fn new<S1, S2>(api_key: S1, project: S2) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
    {
        Self {
            api_key: api_key.into(),
            entity: None,
            project: project.into(),
            run_name: None,
        }
    }

    pub fn entity(mut self, entity: impl Into<String>) -> Self {
        self.entity = Some(entity.into());

        self
    }

    pub fn run_name(mut self, run_name: impl Into<String>) -> Self {
        self.run_name = Some(run_name.into());

        self
    }
}

/// Initialize a WandB logger following Burn's async logger pattern.
pub async fn init_logger(config: WandbLoggerConfig) -> AsyncLogger<LogData> {
    let wandb = WandB::new(BackendOptions::new(config.api_key));

    let mut run_info = RunInfo::new(config.project);

    if let Some(entity) = config.entity {
        run_info = run_info.entity(entity);
    }

    if let Some(run_name) = config.run_name {
        run_info = run_info.name(run_name);
    }

    let run = wandb
        .new_run(run_info.build().expect("wandb run info should be valid"))
        .await
        .expect("wandb run creation should succeed");

    AsyncLogger::new(WandbLogger::new(run))
}

pub fn init_logger_blocking(config: WandbLoggerConfig) -> AsyncLogger<LogData> {
    // Keep the runtime alive for the entire process so wandb-rs background tasks don't get dropped.
    wandb_runtime().block_on(init_logger(config))
}

pub async fn init_metric_logger(config: WandbLoggerConfig) -> WandbLogger {
    let wandb = WandB::new(BackendOptions::new(config.api_key));

    let mut run_info = RunInfo::new(config.project);

    if let Some(entity) = config.entity {
        run_info = run_info.entity(entity);
    }

    if let Some(run_name) = config.run_name {
        run_info = run_info.name(run_name);
    }

    let run = wandb
        .new_run(run_info.build().expect("wandb run info should be valid"))
        .await
        .expect("wandb run creation should succeed");

    WandbLogger::new(run)
}

pub fn init_metric_logger_blocking(config: WandbLoggerConfig) -> WandbLogger {
    // Keep the runtime alive for the entire process so wandb-rs background tasks don't get dropped.
    wandb_runtime().block_on(init_metric_logger(config))
}

pub struct WandbLogger {
    run: Arc<Mutex<Option<Run>>>,
    runtime: Runtime,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    global_step: u64,
}

struct CudaMetricEntry {
    index: usize,
    mem_used_gb: f64,
    mem_total_gb: f64,
    util_pct: f64,
}

fn parse_cuda_metrics(formatted: &str) -> Vec<CudaMetricEntry> {
    let mut entries = Vec::new();
    let mut rest = formatted;
    let memory_marker = " - Memory ";
    let usage_marker = " Gb - Usage ";

    loop {
        let pos = match rest.find("GPU #") {
            Some(pos) => pos,
            None => break,
        };
        rest = &rest[pos + 5..];

        let memory_pos = match rest.find(memory_marker) {
            Some(pos) => pos,
            None => break,
        };
        let index_str = rest[..memory_pos].trim();
        let index: usize = match index_str.parse() {
            Ok(index) => index,
            Err(_) => {
                rest = &rest[memory_pos + memory_marker.len()..];
                continue;
            }
        };

        let after_memory = &rest[memory_pos + memory_marker.len()..];
        let usage_pos = match after_memory.find(usage_marker) {
            Some(pos) => pos,
            None => break,
        };

        let memory_str = after_memory[..usage_pos].trim();
        let mut parts = memory_str.split('/');
        let mem_used_gb: f64 = match parts.next().and_then(|val| val.trim().parse().ok()) {
            Some(value) => value,
            None => {
                rest = &after_memory[usage_pos + usage_marker.len()..];
                continue;
            }
        };
        let mem_total_gb: f64 = match parts.next().and_then(|val| val.trim().parse().ok()) {
            Some(value) => value,
            None => {
                rest = &after_memory[usage_pos + usage_marker.len()..];
                continue;
            }
        };

        let after_usage = &after_memory[usage_pos + usage_marker.len()..];
        let percent_pos = match after_usage.find('%') {
            Some(pos) => pos,
            None => break,
        };
        let util_pct: f64 = match after_usage[..percent_pos].trim().parse() {
            Ok(value) => value,
            Err(_) => {
                rest = &after_usage[percent_pos + 1..];
                continue;
            }
        };

        entries.push(CudaMetricEntry {
            index,
            mem_used_gb,
            mem_total_gb,
            util_pct,
        });

        rest = &after_usage[percent_pos + 1..];
    }

    entries
}

impl WandbLogger {
    fn new(run: Run) -> Self {
        Self {
            run: Arc::new(Mutex::new(Some(run))),
            runtime: Runtime::new().expect("tokio runtime should be created"),
            metric_definitions: HashMap::new(),
            global_step: 0,
        }
    }
}

impl Drop for WandbLogger {
    fn drop(&mut self) {
        let run = Arc::clone(&self.run);
        let handle = self.runtime.handle().clone();
        let _ = std::thread::spawn(move || {
            handle.block_on(async move {
                let mut guard = run.lock().await;
                if let Some(run) = guard.take() {
                    if let Err(err) = run.finish().await {
                        warn!("Failed to finish wandb run: {err}");
                    }
                }
            });
        })
        .join();
    }
}

impl Logger<LogData> for WandbLogger {
    fn log(&mut self, item: LogData) {
        let run = Arc::clone(&self.run);

        self.runtime.spawn(async move {
            let guard = run.lock().await;
            if let Some(run) = guard.as_ref() {
                run.log(item).await;
            }
        });
    }
}

impl MetricLogger for WandbLogger {
    fn log(&mut self, update: MetricsUpdate, _epoch: usize, split: Split, tag: Option<Arc<String>>) {
        self.global_step += 1;

        let mut log = LogData::new();
        log.insert("_step", self.global_step);
        // log.insert("epoch", epoch as u64);

        let metric_prefix = match tag.as_deref() {
            Some(tag) => {
                let tag = tag.trim().replace(' ', "-").to_lowercase();
                format!("{split}/{tag}/")
            }
            None => format!("{split}/"),
        };

        if let Some(tag) = tag.as_deref() {
            log.insert("tag", tag.to_string());
        }

        for entry in update.entries {
            let metric_id = entry.metric_id;
            let metric_name = self
                .metric_definitions
                .get(&metric_id)
                .map(|definition| definition.name.as_str())
                .unwrap_or("metric");
            let name = format!("{metric_prefix}{metric_name}");
            let SerializedEntry { formatted, serialized } = entry.serialized_entry;
            let value = if formatted.is_empty() {
                serialized.as_str()
            } else {
                formatted.as_str()
            };
            log.insert(name, value);

            if metric_name == "Cuda" {
                let parsed = parse_cuda_metrics(value);
                for entry in parsed {
                    let base = format!("{metric_prefix}cuda/gpu{}/", entry.index);
                    log.insert(format!("{base}mem_used_gb"), entry.mem_used_gb);
                    log.insert(format!("{base}mem_total_gb"), entry.mem_total_gb);
                    if entry.mem_total_gb > 0.0 {
                        log.insert(
                            format!("{base}mem_used_pct"),
                            entry.mem_used_gb / entry.mem_total_gb * 100.0,
                        );
                    }
                    log.insert(format!("{base}util_pct"), entry.util_pct);
                }
            }
        }

        for entry in update.entries_numeric {
            let name = self
                .metric_definitions
                .get(&entry.entry.metric_id)
                .map(|definition| definition.name.as_str())
                .unwrap_or("metric");
            let name = format!("{metric_prefix}{name}");

            match entry.numeric_entry {
                NumericEntry::Value(value) => {
                    log.insert(name, value);
                }
                NumericEntry::Aggregated { aggregated_value, .. } => {
                    log.insert(name, aggregated_value);
                }
            }
        }

        let run = Arc::clone(&self.run);
        self.runtime.spawn(async move {
            let guard = run.lock().await;
            if let Some(run) = guard.as_ref() {
                run.log(log).await;
            }
        });
    }

    fn read_numeric(
        &mut self,
        _name: &str,
        _epoch: usize,
        _split: Split,
    ) -> Result<Vec<NumericEntry>, String> {
        Ok(Vec::new())
    }

    fn log_metric_definition(&mut self, definition: MetricDefinition) {
        self.metric_definitions
            .insert(definition.metric_id.clone(), definition);
    }

    fn log_epoch_summary(&mut self, _summary: EpochSummary) {}
}
