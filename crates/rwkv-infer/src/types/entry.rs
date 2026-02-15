use crate::config::SamplingConfig;
use tokio::sync::mpsc;
use uuid::Uuid;

pub type EntryId = Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferEntryState {
    Waiting,
    RunningPrefill,
    RunningDecode,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug)]
pub struct InferEntry {
    pub entry_id: EntryId,
    pub state: InferEntryState,

    pub batch_index: Option<usize>,

    pub input_text: String,

    pub input_token_ids: Vec<i32>,
    pub prefill_padded_token_ids: Vec<i32>,
    pub prefill_pad_len: usize,
    pub prefill_chunk_cursor: usize,

    pub generated_token_ids: Vec<i32>,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,

    pub stream_tx: Option<mpsc::Sender<EngineEvent>>,
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    Text(String),
    Done,
    Error(String),
}

impl InferEntry {
    pub fn new(entry_id: EntryId, input_text: String, sampling: SamplingConfig) -> Self {
        Self {
            entry_id,
            state: InferEntryState::Waiting,
            batch_index: None,
            input_text,
            input_token_ids: Vec::new(),
            prefill_padded_token_ids: Vec::new(),
            prefill_pad_len: 0,
            prefill_chunk_cursor: 0,
            generated_token_ids: Vec::new(),
            max_new_tokens: sampling.max_new_tokens,
            sampling,
            stream_tx: None,
        }
    }
}
