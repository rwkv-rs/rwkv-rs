use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type InferenceRequestId = Uuid;
pub type EntryId = InferenceRequestId;

#[derive(Clone, Debug, PartialEq)]
pub struct OutputTokenCandidate {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputToken {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<OutputTokenCandidate>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamDelta {
    pub text: String,
    pub tokens: Vec<OutputToken>,
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    Output(StreamDelta),
    Done(FinishMetadata),
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinishMetadata {
    pub reason: FinishReason,
    pub matched_stop_suffix: Option<String>,
    pub matched_stop_suffix_index: Option<usize>,
    pub max_new_tokens: usize,
    pub generated_tokens: usize,
    pub timings_ms: Option<TimingBreakdownMs>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingBreakdownMs {
    pub validate_ms: Option<u64>,
    pub tokenize_ms: Option<u64>,
    pub queue_wait_ms: Option<u64>,
    pub schedule_wait_ms: Option<u64>,
    pub prefill_first_ms: Option<u64>,
    pub first_emit_ms: Option<u64>,
    pub prefill_total_ms: Option<u64>,
    pub decode_total_ms: Option<u64>,
    pub request_total_ms: Option<u64>,
}
