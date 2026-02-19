use tokio::sync::mpsc;
use uuid::Uuid;

use crate::types::SamplingConfig;

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

#[derive(Clone, Debug)]
pub enum EngineEvent {
    Text(String),
    Done,
    Error(String),
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
    pub generated_bytes: Vec<u8>,
    pub emitted_byte_len: usize,
    pub utf8_carry: Vec<u8>,

    pub stop_suffixes: Vec<Vec<u8>>,
    pub max_stop_suffix_len: usize,
    pub matched_stop_suffix_len: usize,

    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,

    pub stream_tx: Option<mpsc::Sender<EngineEvent>>,
}

impl InferEntry {
    pub fn new(
        entry_id: EntryId,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
    ) -> Self {
        let stop_suffixes: Vec<Vec<u8>> = stop_suffixes
            .into_iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.into_bytes())
            .collect();
        let max_stop_suffix_len = stop_suffixes.iter().map(|s| s.len()).max().unwrap_or(0);

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
            generated_bytes: Vec::new(),
            emitted_byte_len: 0,
            utf8_carry: Vec::new(),
            stop_suffixes,
            max_stop_suffix_len,
            matched_stop_suffix_len: 0,
            max_new_tokens: sampling.max_new_tokens,
            sampling,
            stream_tx: None,
        }
    }

    pub fn stop_trunc_len(&self) -> usize {
        self.generated_bytes
            .len()
            .saturating_sub(self.matched_stop_suffix_len)
    }

    pub fn emit_stream_text(&mut self) -> (String, bool) {
        if self.max_stop_suffix_len > 0 {
            self.matched_stop_suffix_len =
                match_stop_suffix_len(&self.generated_bytes, &self.stop_suffixes);
            if self.matched_stop_suffix_len > 0 {
                let trunc_len = self.stop_trunc_len();
                let text = self.flush_stream_text_until(trunc_len, false);
                return (text, true);
            }
        }

        let hold_len = self.max_stop_suffix_len.saturating_sub(1);
        let emit_limit = self.generated_bytes.len().saturating_sub(hold_len);
        let text = self.flush_stream_text_until(emit_limit, false);
        (text, false)
    }

    pub fn flush_stream_text_until(&mut self, emit_limit: usize, flush_lossy: bool) -> String {
        if self.emitted_byte_len >= emit_limit {
            if flush_lossy && !self.utf8_carry.is_empty() {
                let out = String::from_utf8_lossy(&self.utf8_carry).into_owned();
                self.utf8_carry.clear();
                return out;
            }
            return String::new();
        }

        let new_bytes = &self.generated_bytes[self.emitted_byte_len..emit_limit];
        let mut buf = Vec::with_capacity(self.utf8_carry.len() + new_bytes.len());
        buf.extend_from_slice(&self.utf8_carry);
        buf.extend_from_slice(new_bytes);

        self.emitted_byte_len = emit_limit;

        if flush_lossy {
            self.utf8_carry.clear();
            return String::from_utf8_lossy(&buf).into_owned();
        }

        match std::str::from_utf8(&buf) {
            Ok(s) => {
                self.utf8_carry.clear();
                s.to_string()
            }
            Err(err) => {
                let valid = err.valid_up_to();
                if valid == 0 {
                    self.utf8_carry.clear();
                    return String::from_utf8_lossy(&buf).into_owned();
                }

                let prefix = std::str::from_utf8(&buf[..valid])
                    .unwrap_or_default()
                    .to_string();
                if err.error_len().is_none() {
                    self.utf8_carry = buf[valid..].to_vec();
                    prefix
                } else {
                    self.utf8_carry.clear();
                    let rest = String::from_utf8_lossy(&buf[valid..]).into_owned();
                    prefix + &rest
                }
            }
        }
    }
}

fn match_stop_suffix_len(output: &[u8], suffixes: &[Vec<u8>]) -> usize {
    suffixes
        .iter()
        .filter(|suffix| !suffix.is_empty() && output.ends_with(suffix.as_slice()))
        .map(|suffix| suffix.len())
        .max()
        .unwrap_or(0)
}
