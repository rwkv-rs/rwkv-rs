use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::engine::{EngineHandle, SubmitOutput};
use crate::types::SamplingConfig;

pub mod builder;
pub mod runtime_manager;

pub use runtime_manager::{
    ModelEngineFactory, ModelsReloadPatch, ModelsReloadResult, RuntimeManager,
};

#[derive(Clone)]
pub struct ModelRuntimeGroup {
    engines: Arc<Vec<Arc<EngineHandle>>>,
    next_index: Arc<AtomicUsize>,
}

impl ModelRuntimeGroup {
    pub fn new(model_name: String, engines: Vec<Arc<EngineHandle>>) -> crate::Result<Self> {
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

    fn select_engine(&self) -> Arc<EngineHandle> {
        let index = self.next_index.fetch_add(1, Ordering::Relaxed) % self.engines.len();
        self.engines[index].clone()
    }
}

#[derive(Clone)]
pub struct Service {
    model_groups: Arc<HashMap<String, ModelRuntimeGroup>>,
}

impl Service {
    pub fn new(model_groups: HashMap<String, ModelRuntimeGroup>) -> crate::Result<Self> {
        if model_groups.is_empty() {
            return Err(crate::Error::BadRequest(
                "at least one model group is required".to_string(),
            ));
        }

        Ok(Self {
            model_groups: Arc::new(model_groups),
        })
    }

    pub fn model_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.model_groups.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn clone_model_groups(&self) -> HashMap<String, ModelRuntimeGroup> {
        self.model_groups.as_ref().clone()
    }

    pub async fn submit_text(
        &self,
        model_name: &str,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        stream: bool,
    ) -> crate::Result<SubmitOutput> {
        self.submit_text_with_trace(
            model_name,
            input_text,
            sampling,
            stop_suffixes,
            stream,
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
        stream: bool,
        validate_ms: Option<u64>,
    ) -> crate::Result<SubmitOutput> {
        #[cfg(feature = "trace")]
        tracing::info!(
            target: "rwkv.infer",
            model = %model_name,
            stream,
            max_new_tokens = sampling.max_new_tokens,
            validate_ms,
            "dispatch submit request"
        );

        let group = self.model_groups.get(model_name).ok_or_else(|| {
            crate::Error::BadRequest(format!(
                "unknown model_name: {model_name}. available: {:?}",
                self.model_names()
            ))
        })?;

        group
            .select_engine()
            .submit_text(input_text, sampling, stop_suffixes, stream, validate_ms)
            .await
    }
}
