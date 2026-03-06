use std::time::Instant;

use tokio::sync::mpsc;
#[cfg(test)]
use uuid::Uuid;

#[cfg(test)]
use crate::inference_core::OutputTokenCandidate;
use crate::inference_core::{
    EngineEvent, EntryId, OutputToken, SamplingConfig, StreamDelta, TokenLogprobsConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveRequestState {
    Waiting,
    RunningPrefill,
    RunningDecode,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StopMatch {
    pub index: usize,
    pub len: usize,
}

#[derive(Debug)]
pub struct ActiveRequest {
    pub entry_id: EntryId,
    pub state: ActiveRequestState,

    pub batch_index: Option<usize>,

    pub input_text: String,

    pub input_token_ids: Vec<i32>,
    pub prefill_padded_token_ids: Vec<i32>,
    pub prefill_pad_len: usize,
    pub prefill_chunk_cursor: usize,

    pub generated_token_ids: Vec<i32>,
    pub generated_bytes: Vec<u8>,
    pub generated_tokens: Vec<OutputToken>,
    pub emitted_token_count: usize,
    pub emitted_byte_len: usize,

    pub stop_suffixes: Vec<Vec<u8>>,
    pub max_stop_suffix_len: usize,
    pub matched_stop_suffix: Option<StopMatch>,

    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub token_logprobs: Option<TokenLogprobsConfig>,

    pub stream_tx: Option<mpsc::Sender<EngineEvent>>,

    pub validate_ms: Option<u64>,
    pub tokenize_ms: Option<u64>,
    pub submitted_at: Option<Instant>,
    pub runtime_received_at: Option<Instant>,
    pub enqueued_at: Option<Instant>,
    pub scheduled_at: Option<Instant>,
    pub first_prefill_at: Option<Instant>,
    pub last_prefill_at: Option<Instant>,
    pub first_decode_at: Option<Instant>,
    pub last_decode_at: Option<Instant>,
    pub first_emit_at: Option<Instant>,
}

impl ActiveRequest {
    pub fn new(
        entry_id: EntryId,
        input_text: String,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        token_logprobs: Option<TokenLogprobsConfig>,
    ) -> Self {
        let stop_suffixes: Vec<Vec<u8>> = stop_suffixes
            .into_iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.into_bytes())
            .collect();
        let max_stop_suffix_len = stop_suffixes.iter().map(|s| s.len()).max().unwrap_or(0);

        Self {
            entry_id,
            state: ActiveRequestState::Waiting,
            batch_index: None,
            input_text,
            input_token_ids: Vec::new(),
            prefill_padded_token_ids: Vec::new(),
            prefill_pad_len: 0,
            prefill_chunk_cursor: 0,
            generated_token_ids: Vec::new(),
            generated_bytes: Vec::new(),
            generated_tokens: Vec::new(),
            emitted_token_count: 0,
            emitted_byte_len: 0,
            stop_suffixes,
            max_stop_suffix_len,
            matched_stop_suffix: None,
            max_new_tokens: sampling.max_new_tokens,
            sampling,
            token_logprobs,
            stream_tx: None,
            validate_ms: None,
            tokenize_ms: None,
            submitted_at: None,
            runtime_received_at: None,
            enqueued_at: None,
            scheduled_at: None,
            first_prefill_at: None,
            last_prefill_at: None,
            first_decode_at: None,
            last_decode_at: None,
            first_emit_at: None,
        }
    }

    pub fn stop_trunc_len(&self) -> usize {
        let matched_len = self.matched_stop_suffix.map(|m| m.len).unwrap_or(0);
        self.generated_bytes.len().saturating_sub(matched_len)
    }

    pub fn push_generated_token(&mut self, token: OutputToken, token_id: i32) {
        self.generated_token_ids.push(token_id);
        self.generated_bytes.extend_from_slice(&token.bytes);
        self.generated_tokens.push(token);
    }

    pub fn emit_stream_delta(&mut self) -> (StreamDelta, Option<StopMatch>) {
        if self.max_stop_suffix_len > 0 {
            self.matched_stop_suffix =
                match_stop_suffix(&self.generated_bytes, &self.stop_suffixes);
            if let Some(matched) = self.matched_stop_suffix {
                let trunc_len = self.stop_trunc_len();
                let delta = self.flush_stream_delta_until(trunc_len, true);
                return (delta, Some(matched));
            }
        }
        self.matched_stop_suffix = None;

        let hold_len = self.max_stop_suffix_len.saturating_sub(1);
        let emit_limit = self.generated_bytes.len().saturating_sub(hold_len);
        let delta = self.flush_stream_delta_until(emit_limit, false);
        (delta, None)
    }

    pub fn flush_stream_delta_until(
        &mut self,
        emit_limit: usize,
        allow_partial_token: bool,
    ) -> StreamDelta {
        if self.emitted_byte_len >= emit_limit {
            return StreamDelta::default();
        }

        let mut full_token_limit = self.emitted_token_count;
        let mut byte_cursor = self.emitted_byte_len;
        while let Some(token) = self.generated_tokens.get(full_token_limit) {
            let next_byte_cursor = byte_cursor + token.bytes.len();
            if next_byte_cursor > emit_limit {
                break;
            }
            byte_cursor = next_byte_cursor;
            full_token_limit += 1;
        }

        let candidate_tokens = &self.generated_tokens[self.emitted_token_count..full_token_limit];
        let emit_full_count = longest_valid_utf8_token_prefix(candidate_tokens);
        let emit_tokens_end = self.emitted_token_count + emit_full_count;

        let mut emitted_tokens =
            self.generated_tokens[self.emitted_token_count..emit_tokens_end].to_vec();
        let mut emitted_text = decode_tokens_text(&emitted_tokens);

        self.emitted_token_count = emit_tokens_end;
        self.emitted_byte_len += emitted_tokens
            .iter()
            .map(|token| token.bytes.len())
            .sum::<usize>();

        if allow_partial_token && self.emitted_byte_len < emit_limit {
            let partial_len = emit_limit - self.emitted_byte_len;
            if let Some(token) = self.generated_tokens.get(self.emitted_token_count) {
                let partial_bytes = &token.bytes[..partial_len.min(token.bytes.len())];
                let valid_prefix_len = longest_valid_utf8_prefix_len(partial_bytes);
                if valid_prefix_len > 0 {
                    let bytes = partial_bytes[..valid_prefix_len].to_vec();
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    emitted_text.push_str(&text);
                    emitted_tokens.push(OutputToken {
                        token: text,
                        bytes,
                        logprob: token.logprob,
                        top_logprobs: token.top_logprobs.clone(),
                    });
                    self.emitted_byte_len += valid_prefix_len;
                }
            }
        }

        StreamDelta {
            text: emitted_text,
            tokens: emitted_tokens,
        }
    }
}

fn longest_valid_utf8_token_prefix(tokens: &[OutputToken]) -> usize {
    if tokens.is_empty() {
        return 0;
    }

    let mut buf = Vec::new();
    let mut last_valid = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        buf.extend_from_slice(&token.bytes);
        match std::str::from_utf8(&buf) {
            Ok(_) => last_valid = index + 1,
            Err(err) if err.error_len().is_none() => {}
            Err(_) => {
                last_valid = index + 1;
                break;
            }
        }
    }
    last_valid
}

fn decode_tokens_text(tokens: &[OutputToken]) -> String {
    if tokens.is_empty() {
        return String::new();
    }

    let mut bytes = Vec::new();
    for token in tokens {
        bytes.extend_from_slice(&token.bytes);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn longest_valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(err) => err.valid_up_to(),
    }
}

fn match_stop_suffix(output: &[u8], suffixes: &[Vec<u8>]) -> Option<StopMatch> {
    let mut best: Option<StopMatch> = None;
    for (index, suffix) in suffixes.iter().enumerate() {
        if suffix.is_empty() || !output.ends_with(suffix.as_slice()) {
            continue;
        }
        let candidate = StopMatch {
            index,
            len: suffix.len(),
        };
        match best {
            None => best = Some(candidate),
            Some(current) => {
                if candidate.len > current.len
                    || (candidate.len == current.len && candidate.index < current.index)
                {
                    best = Some(candidate);
                }
            }
        }
    }
    best
}

pub type InferEntry = ActiveRequest;
pub type InferEntryState = ActiveRequestState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_stop_suffix_prefers_longer_suffix() {
        let suffixes = vec![b"END".to_vec(), b"THE END".to_vec()];
        let output = b"This is THE END";
        let matched = match_stop_suffix(output, &suffixes).expect("must match");
        assert_eq!(matched.index, 1);
        assert_eq!(matched.len, 7);
    }

    #[test]
    fn match_stop_suffix_prefers_lower_index_on_tie() {
        let suffixes = vec![b"stop".to_vec(), b"stop".to_vec()];
        let output = b"please stop";
        let matched = match_stop_suffix(output, &suffixes).expect("must match");
        assert_eq!(matched.index, 0);
        assert_eq!(matched.len, 4);
    }

    #[test]
    fn flush_stream_delta_emits_partial_visible_prefix_on_stop() {
        let sampling = SamplingConfig::default();
        let mut entry = ActiveRequest::new(
            Uuid::new_v4(),
            "prompt".to_string(),
            sampling,
            vec!["END".to_string()],
            Some(TokenLogprobsConfig {
                top_logprobs: 2,
                candidate_token_ids: None,
            }),
        );

        entry.push_generated_token(
            OutputToken {
                token: " THE END".to_string(),
                bytes: b" THE END".to_vec(),
                logprob: Some(-0.5),
                top_logprobs: vec![OutputTokenCandidate {
                    token: " THE END".to_string(),
                    bytes: b" THE END".to_vec(),
                    logprob: -0.5,
                }],
            },
            42,
        );

        let (delta, matched) = entry.emit_stream_delta();
        assert_eq!(matched.expect("stop must match").index, 0);
        assert_eq!(delta.text, " THE ");
        assert_eq!(delta.tokens.len(), 1);
        assert_eq!(delta.tokens[0].token, " THE ");
        assert_eq!(delta.tokens[0].bytes, b" THE ".to_vec());
    }
}
