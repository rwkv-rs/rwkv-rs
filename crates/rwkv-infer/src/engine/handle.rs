use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::types::{EngineEvent, SamplingConfig};


pub type EntryId = Uuid;

#[derive(Debug)]
pub enum EngineCommand {
    SubmitText {
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        stream: bool,
        reply: oneshot::Sender<SubmitOutput>,
    },
    Cancel {
        entry_id: EntryId,
    },
}

#[derive(Debug)]
pub enum SubmitOutput {
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
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    pub fn new(tx: mpsc::Sender<EngineCommand>) -> Self {
        Self { tx }
    }

    pub async fn submit_text(
        &self,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        stream: bool,
    ) -> crate::Result<SubmitOutput> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::SubmitText {
                input_text,
                sampling,
                stop_suffixes,
                stream,
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
            .send(EngineCommand::Cancel { entry_id })
            .await
            .map_err(|e| crate::Error::Internal(format!("engine channel closed: {e}")))?;
        Ok(())
    }
}
