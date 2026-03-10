use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::inference_core::{InferenceSubmitHandle, InferenceSubmitResult};
use crate::inference_core::{RequestedTokenLogprobsConfig, SamplingConfig};

#[derive(Clone)]
pub struct LoadedModelGroup {
    engines: Arc<Vec<Arc<InferenceSubmitHandle>>>,
    next_index: Arc<AtomicUsize>,
}

impl LoadedModelGroup {
    pub fn new(
        model_name: String,
        engines: Vec<Arc<InferenceSubmitHandle>>,
    ) -> crate::Result<Self> {
        if engines.is_empty() {
            return Err(crate::Error::BadRequest(format!(
                "model {model_name} has no running engine"
            )));
        }
        Ok(Self {
            engines: Arc::new(engines),
            next_index: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn select_engine(&self) -> Arc<InferenceSubmitHandle> {
        let index = self.next_index.fetch_add(1, Ordering::Relaxed) % self.engines.len();
        self.engines[index].clone()
    }
}

#[derive(Clone)]
pub struct ModelRequestRouter {
    model_groups: Arc<HashMap<String, LoadedModelGroup>>,
    model_vocab_sizes: Arc<HashMap<String, usize>>,
}

impl ModelRequestRouter {
    pub fn new(
        model_groups: HashMap<String, LoadedModelGroup>,
        model_vocab_sizes: HashMap<String, usize>,
    ) -> crate::Result<Self> {
        if model_groups.is_empty() {
            return Err(crate::Error::BadRequest(
                "at least one model group is required".to_string(),
            ));
        }

        Ok(Self {
            model_groups: Arc::new(model_groups),
            model_vocab_sizes: Arc::new(model_vocab_sizes),
        })
    }

    pub fn model_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.model_groups.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn clone_model_groups(&self) -> HashMap<String, LoadedModelGroup> {
        self.model_groups.as_ref().clone()
    }

    pub fn clone_model_vocab_sizes(&self) -> HashMap<String, usize> {
        self.model_vocab_sizes.as_ref().clone()
    }

    pub fn model_vocab_size(&self, model_name: &str) -> Option<usize> {
        self.model_vocab_sizes.get(model_name).copied()
    }

    pub async fn submit_text(
        &self,
        model_name: &str,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        requested_token_logprobs: Option<RequestedTokenLogprobsConfig>,
    ) -> crate::Result<InferenceSubmitResult> {
        self.submit_text_with_trace(
            model_name,
            input_text,
            sampling,
            stop_suffixes,
            requested_token_logprobs,
            None,
        )
        .await
    }

    pub async fn submit_text_with_trace(
        &self,
        model_name: &str,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        requested_token_logprobs: Option<RequestedTokenLogprobsConfig>,
        validate_ms: Option<u64>,
    ) -> crate::Result<InferenceSubmitResult> {
        let group = self.model_groups.get(model_name).ok_or_else(|| {
            crate::Error::BadRequest(format!(
                "unknown model_name: {model_name}. available: {:?}",
                self.model_names()
            ))
        })?;

        group
            .select_engine()
            .submit_text(
                input_text,
                sampling,
                stop_suffixes,
                requested_token_logprobs,
                validate_ms,
            )
            .await
    }
}
