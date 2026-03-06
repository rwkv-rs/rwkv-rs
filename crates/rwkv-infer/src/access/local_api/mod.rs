use std::sync::Arc;

use crate::inference_core::SamplingConfig;
use crate::inference_core::{
    InferenceSubmitHandle as EngineHandle, InferenceSubmitResult as SubmitOutput,
};

#[derive(Clone)]
pub struct LocalInferenceClient {
    engine: Arc<EngineHandle>,
}

impl LocalInferenceClient {
    pub fn new(engine: Arc<EngineHandle>) -> Self {
        Self { engine }
    }

    pub async fn completions_text(
        &self,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        stream: bool,
    ) -> crate::Result<SubmitOutput> {
        self.engine
            .submit_text(input_text, sampling, stop_suffixes, None, stream, None)
            .await
    }
}

pub type RwkvInferClient = LocalInferenceClient;
