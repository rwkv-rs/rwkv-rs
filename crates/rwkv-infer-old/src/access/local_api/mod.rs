use std::sync::Arc;

use tokio::sync::mpsc;

use crate::{
    access::http_api::collect_stream_output,
    inference_core::{EngineEvent, InferenceSubmitHandle, InferenceSubmitResult, SamplingConfig},
};

#[derive(Clone)]
pub struct LocalInferenceClient {
    engine: Arc<InferenceSubmitHandle>,
}

impl LocalInferenceClient {
    pub fn new(engine: Arc<InferenceSubmitHandle>) -> Self {
        Self { engine }
    }

    pub async fn submit_text_receiver(
        &self,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
    ) -> crate::Result<mpsc::Receiver<EngineEvent>> {
        let submit = self
            .engine
            .submit_text(input_text, sampling, stop_suffixes, None, None, None)
            .await?;
        expect_submit_receiver(submit)
    }

    pub async fn collect_text(
        &self,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
    ) -> crate::Result<String> {
        let mut rx = self
            .submit_text_receiver(input_text, sampling, stop_suffixes)
            .await?;
        Ok(collect_stream_output(&mut rx).await?.text)
    }
}

fn expect_submit_receiver(
    submit: InferenceSubmitResult,
) -> crate::Result<mpsc::Receiver<EngineEvent>> {
    match submit {
        InferenceSubmitResult::Receiver { rx, .. } => Ok(rx),
        InferenceSubmitResult::Error { message, .. } => Err(crate::Error::bad_request(message)),
    }
}

pub type RwkvInferClient = LocalInferenceClient;
