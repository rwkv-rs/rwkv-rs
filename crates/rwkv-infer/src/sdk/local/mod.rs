use std::{path::PathBuf, sync::Arc};

use crate::{
    dtos::{
        admin::models::reload::{ModelsReloadReq, ModelsReloadResp},
        chat::completions::{ChatCompletionResp, ChatCompletionsReq},
        completions::{CompletionsReq, CompletionsResp},
        health::HealthResp,
        models::ModelsResp,
    },
    services::{
        QueueMap,
        QueueMapBuilder,
        ServiceResult,
        SharedQueueMap,
        chat::completions::ChatCompletionRun,
        completions::CompletionRun,
        health::GpuMetricsCache,
        select_model_queue,
        shared_queue_map,
    },
};

#[derive(Clone)]
pub struct LocalClient {
    queues: SharedQueueMap,
    gpu_metrics: Arc<GpuMetricsCache>,
    reload_lock: Option<Arc<tokio::sync::Mutex<()>>>,
    infer_cfg_path: Option<PathBuf>,
    build_queues: Option<QueueMapBuilder>,
}

impl LocalClient {
    pub fn new(queues: QueueMap) -> Self {
        Self {
            queues: shared_queue_map(queues),
            gpu_metrics: Arc::new(GpuMetricsCache::new()),
            reload_lock: None,
            infer_cfg_path: None,
            build_queues: None,
        }
    }

    pub fn from_shared(queues: SharedQueueMap) -> Self {
        Self {
            queues,
            gpu_metrics: Arc::new(GpuMetricsCache::new()),
            reload_lock: None,
            infer_cfg_path: None,
            build_queues: None,
        }
    }

    pub fn with_reload_support(
        mut self,
        infer_cfg_path: PathBuf,
        build_queues: QueueMapBuilder,
    ) -> Self {
        self.reload_lock = Some(Arc::new(tokio::sync::Mutex::new(())));
        self.infer_cfg_path = Some(infer_cfg_path);
        self.build_queues = Some(build_queues);
        self
    }

    pub fn shared_queues(&self) -> SharedQueueMap {
        Arc::clone(&self.queues)
    }

    pub fn health(&self) -> HealthResp {
        let queues = self
            .queues
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        crate::services::health::health(&queues, self.gpu_metrics.as_ref())
    }

    pub fn models(&self) -> ModelsResp {
        let queues = self
            .queues
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        crate::services::models::models(&queues)
    }

    pub async fn completions_run(&self, req: CompletionsReq) -> ServiceResult<CompletionRun> {
        let handle = {
            let queues = self
                .queues
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            select_model_queue(&queues, &req.model)?
        };
        crate::services::completions::completions(handle, req).await
    }

    pub async fn completions(&self, req: CompletionsReq) -> ServiceResult<CompletionsResp> {
        let run = self.completions_run(req).await?;
        Ok(run.collect().await)
    }

    pub async fn chat_completions_run(
        &self,
        req: ChatCompletionsReq,
    ) -> ServiceResult<ChatCompletionRun> {
        let handle = {
            let queues = self
                .queues
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            select_model_queue(&queues, &req.model)?
        };
        crate::services::chat::completions::chat_completions(handle, req).await
    }

    pub async fn chat_completions(
        &self,
        req: ChatCompletionsReq,
    ) -> ServiceResult<ChatCompletionResp> {
        let run = self.chat_completions_run(req).await?;
        Ok(run.collect().await)
    }

    pub async fn reload_models(&self, req: ModelsReloadReq) -> ServiceResult<ModelsReloadResp> {
        crate::services::admin::models::reload::reload_models(
            &self.queues,
            self.reload_lock.as_ref(),
            self.infer_cfg_path.as_deref(),
            self.build_queues.as_ref(),
            req,
        )
        .await
    }
}

pub type RwkvInferClient = LocalClient;
