use std::{collections::HashMap, sync::{Arc, OnceLock}};
use burn::train::{
    logger::{AsyncLogger, Logger, MetricLogger},
    metric::{MetricDefinition, MetricId, NumericEntry},
    metric::store::{EpochSummary, MetricsUpdate, Split},
};
use tokio::runtime::Runtime;
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
    run: Arc<Run>,
    runtime: Runtime,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    global_step: u64,
}

impl WandbLogger {
    fn new(run: Run) -> Self {
        Self {
            run: Arc::new(run),
            runtime: Runtime::new().expect("tokio runtime should be created"),
            metric_definitions: HashMap::new(),
            global_step: 0,
        }
    }
}

impl Logger<LogData> for WandbLogger {
    fn log(&mut self, item: LogData) {
        let run = Arc::clone(&self.run);

        self.runtime.spawn(async move {
            run.log(item).await;
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
            run.log(log).await;
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
