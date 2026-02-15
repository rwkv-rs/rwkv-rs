use std::sync::Arc;

use crate::config::SamplingConfig;
use crate::engine::{EngineHandle, SubmitOutput};

#[derive(Clone)]
pub struct RwkvInferClient {
    engine: Arc<EngineHandle>,
}

impl RwkvInferClient {
    pub fn new(engine: Arc<EngineHandle>) -> Self {
        Self { engine }
    }

    pub async fn completions_text(
        &self,
        input_text: String,
        sampling: SamplingConfig,
        stream: bool,
    ) -> crate::Result<SubmitOutput> {
        self.engine.submit_text(input_text, sampling, stream).await
    }
}
