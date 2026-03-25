use std::time::Instant;

use super::{
    ConstraintSpec,
    ConstraintState,
    EntryId,
    SamplingConfig,
    StopMatch,
    StopSuffixMatcher,
    TokenLogprobsConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestPhase {
    Waiting,
    RunningPrefill,
    RunningDecode,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillStepKind {
    WithoutOutput,
    WithOutput,
}

#[derive(Debug)]
pub struct RequestState {
    pub entry_id: EntryId,
    pub phase: RequestPhase,
    pub batch_index: Option<usize>,
    pub input_token_ids: Vec<i32>,
    pub prefill_pad_len: usize,
    pub prefill_chunk_cursor: usize,
    pub generated_token_count: usize,
    pub last_token_id: Option<i32>,
    pub stop_matcher: StopSuffixMatcher,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub constraint_spec: Option<ConstraintSpec>,
    pub constraint: Option<ConstraintState>,
    pub token_logprobs: Option<TokenLogprobsConfig>,
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

impl RequestState {
    pub fn new(
        entry_id: EntryId,
        sampling: SamplingConfig,
        stop_suffixes: Vec<String>,
        constraint_spec: Option<ConstraintSpec>,
        token_logprobs: Option<TokenLogprobsConfig>,
    ) -> Self {
        Self {
            entry_id,
            phase: RequestPhase::Waiting,
            batch_index: None,
            input_token_ids: Vec::new(),
            prefill_pad_len: 0,
            prefill_chunk_cursor: 0,
            generated_token_count: 0,
            last_token_id: None,
            stop_matcher: StopSuffixMatcher::new(stop_suffixes),
            max_new_tokens: sampling.max_new_tokens,
            sampling,
            constraint_spec,
            constraint: None,
            token_logprobs,
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

    pub fn total_prefill_chunks(&self, paragraph_len: usize) -> usize {
        if self.input_token_ids.is_empty() {
            0
        } else {
            self.input_token_ids.len().div_ceil(paragraph_len)
        }
    }

    pub fn next_prefill_step(&self, paragraph_len: usize) -> Option<PrefillStepKind> {
        let total_chunks = self.total_prefill_chunks(paragraph_len);
        if self.prefill_chunk_cursor >= total_chunks {
            None
        } else if self.prefill_chunk_cursor + 1 == total_chunks {
            Some(PrefillStepKind::WithOutput)
        } else {
            Some(PrefillStepKind::WithoutOutput)
        }
    }

    pub fn last_context_token_id(&self) -> Option<i32> {
        self.last_token_id
            .or_else(|| self.input_token_ids.last().copied())
    }

    pub fn on_output_token(&mut self, token_id: i32, token_bytes: &[u8]) -> Option<StopMatch> {
        self.generated_token_count += 1;
        self.last_token_id = Some(token_id);
        self.stop_matcher.push_bytes(token_bytes)
    }

    pub fn hit_max_new_tokens(&self) -> bool {
        self.generated_token_count >= self.max_new_tokens
    }

    pub fn matched_stop_suffix(&self, index: usize) -> Option<String> {
        self.stop_matcher
            .suffix(index)
            .map(|suffix| String::from_utf8_lossy(suffix).into_owned())
    }
}
