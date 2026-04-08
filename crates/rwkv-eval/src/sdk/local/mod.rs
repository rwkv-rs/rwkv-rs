use std::path::PathBuf;

use rwkv_config::validated::eval::FinalEvalConfigBuilder;

use crate::{cores::evaluation, services::ServiceResult};

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
        evaluation::run_evaluation(evaluation::EvaluationRunRequest {
            config: eval_cfg_builder.build_local(),
            datasets_path,
            config_path,
            logs_path,
            runtime_control: None,
        })
        .await
    }
}

pub type RwkvEvalClient = LocalClient;
