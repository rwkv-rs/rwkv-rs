use std::time::Instant;

use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::{EngineEvent, EntryId, SamplingConfig, TokenLogprobsConfig};

#[derive(Debug)]
pub enum InferenceSubmitCommand {
    SubmitText {
        entry_id: EntryId,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        token_logprobs: Option<TokenLogprobsConfig>,
        stream: bool,
        submitted_at: Instant,
        validate_ms: Option<u64>,
        reply: oneshot::Sender<InferenceSubmitResult>,
    },
    Cancel {
        entry_id: EntryId,
    },
}

#[derive(Debug)]
pub enum InferenceSubmitResult {
    Stream {
        entry_id: EntryId,
        rx: mpsc::Receiver<EngineEvent>,
    },
    Done {
        entry_id: EntryId,
        output_text: String,
        token_ids: Vec<i32>,
    },
    Error {
        entry_id: EntryId,
        message: String,
    },
}

#[derive(Clone)]
pub struct InferenceSubmitHandle {
    tx: mpsc::Sender<InferenceSubmitCommand>,
}

impl InferenceSubmitHandle {
    pub fn new(tx: mpsc::Sender<InferenceSubmitCommand>) -> Self {
        Self { tx }
    }

    pub async fn submit_text(
        &self,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        token_logprobs: Option<TokenLogprobsConfig>,
        stream: bool,
        validate_ms: Option<u64>,
    ) -> crate::Result<InferenceSubmitResult> {
        let entry_id = Uuid::new_v4();
        let submitted_at = Instant::now();

        #[cfg(feature = "trace")]
        tracing::info!(
            target: "rwkv.infer",
            stage = "enqueue",
            request_id = %entry_id,
            stream,
            max_new_tokens = sampling.max_new_tokens
        );

        let (reply_tx, reply_rx) = oneshot::channel();
        #[cfg(feature = "trace")]
        let wait_reply_started = Instant::now();
        self.tx
            .send(InferenceSubmitCommand::SubmitText {
                entry_id,
                input_text,
                sampling,
                stop_suffixes,
                token_logprobs,
                stream,
                submitted_at,
                validate_ms,
                reply: reply_tx,
            })
            .await
            .map_err(|e| crate::Error::Internal(format!("engine channel closed: {e}")))?;
        let reply = reply_rx
            .await
            .map_err(|e| crate::Error::Internal(format!("engine reply dropped: {e}")))?;
        #[cfg(feature = "trace")]
        tracing::trace!(
            target: "rwkv.infer",
            request_id = %entry_id,
            wait_reply_ms = wait_reply_started.elapsed().as_millis() as u64,
            "engine submit acknowledged"
        );
        Ok(reply)
    }

    pub async fn cancel(&self, entry_id: EntryId) -> crate::Result<()> {
        #[cfg(feature = "trace")]
        tracing::info!(target: "rwkv.infer", request_id = %entry_id, "cancel requested");
        self.tx
            .send(InferenceSubmitCommand::Cancel { entry_id })
            .await
            .map_err(|e| crate::Error::Internal(format!("engine channel closed: {e}")))?;
        Ok(())
    }
}

pub type EngineCommand = InferenceSubmitCommand;
pub type SubmitOutput = InferenceSubmitResult;
pub type EngineHandle = InferenceSubmitHandle;
