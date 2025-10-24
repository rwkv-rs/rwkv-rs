use std::sync::Arc;

use burn::train::logger::{AsyncLogger, Logger};
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

struct WandbLogger {
    run: Arc<Run>,
    runtime: Runtime,
}

impl WandbLogger {
    fn new(run: Run) -> Self {
        Self {
            run: Arc::new(run),
            runtime: Runtime::new().expect("tokio runtime should be created"),
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
