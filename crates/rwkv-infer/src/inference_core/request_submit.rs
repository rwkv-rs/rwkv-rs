use std::time::Instant;

use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::{ConstraintSpec, EngineEvent, EntryId, RequestedTokenLogprobsConfig, SamplingConfig};

#[derive(Debug)]
pub enum InferenceSubmitCommand {
    SubmitText {
        entry_id: EntryId,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        constraint: Option<ConstraintSpec>,
        requested_token_logprobs: Option<RequestedTokenLogprobsConfig>,
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
    Receiver {
        entry_id: EntryId,
        rx: mpsc::Receiver<EngineEvent>,
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
        constraint: Option<ConstraintSpec>,
        requested_token_logprobs: Option<RequestedTokenLogprobsConfig>,
        validate_ms: Option<u64>,
    ) -> crate::Result<InferenceSubmitResult> {
        let entry_id = Uuid::new_v4();
        let submitted_at = Instant::now();

        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(InferenceSubmitCommand::SubmitText {
                entry_id,
                input_text,
                sampling,
                stop_suffixes,
                constraint,
                requested_token_logprobs,
                submitted_at,
                validate_ms,
                reply: reply_tx,
            })
            .await
            .map_err(|e| crate::Error::Internal(format!("engine channel closed: {e}")))?;
        reply_rx
            .await
            .map_err(|e| crate::Error::Internal(format!("engine reply dropped: {e}")))
    }

    pub async fn cancel(&self, entry_id: EntryId) -> crate::Result<()> {
        self.tx
            .send(InferenceSubmitCommand::Cancel { entry_id })
            .await
            .map_err(|e| crate::Error::Internal(format!("engine channel closed: {e}")))?;
        Ok(())
    }

    pub fn channel(&self) -> mpsc::Sender<InferenceSubmitCommand> {
        self.tx.clone()
    }
}
