use std::path::PathBuf;

use rwkv_config::validated::eval::FinalEvalConfigBuilder;

use crate::services::{ServiceResult, runner};

#[derive(Clone, Debug, Default)]
pub struct LocalClient;

impl LocalClient {
    pub fn new() -> Self {
        Self
    }

    pub async fn run(
        &self,
        eval_cfg_builder: FinalEvalConfigBuilder,
        datasets_path: PathBuf,
        config_path: PathBuf,
        logs_path: PathBuf,
    ) -> ServiceResult<()> {
        runner::run_evaluation(eval_cfg_builder, datasets_path, config_path, logs_path).await
    }
}

pub type RwkvEvalClient = LocalClient;
