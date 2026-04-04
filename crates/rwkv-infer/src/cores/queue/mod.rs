mod detokenize;
mod guided_decode;
pub mod queue_worker;
mod schedule;
mod stats;
mod step;
#[cfg(test)]
mod tests;

use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use indexmap::IndexMap;
use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::mpsc;
use xgrammar::TokenizerInfo;

use self::{
    detokenize::{DetokenizeResult, DetokenizeTask, spawn_detokenize_worker},
    guided_decode::{
        GuidedDecodeResult,
        GuidedDecodeTask,
        GuidedTokenMaskBatchState,
        spawn_guided_decode_worker,
    },
    stats::{QueuePerfHistory, record_perf_sample},
};
use crate::{
    cores::{
        forward::{ModelForward, TokenId, TokenIdLogprobsConfig, sampling::SamplingConfig},
        queue::guided_decode::{GuidedDecodingStatus, build_tokenizer_info_from_vocab},
    },
    dtos::health::QueuePerfStage,
};
pub use self::{guided_decode::GuidedDecodingConfig, stats::snapshot_perf_history};

pub(super) const END_TOKEN_ID: i32 = 0;

pub struct Queue {
    model_forward: Box<dyn ModelForward>,
    max_batch_size: usize,
    paragraph_len: usize,
    guided_vocab_size: usize,
    guided_tokenizer_info: TokenizerInfo,
    guided_token_mask_state: GuidedTokenMaskBatchState,
    items: IndexMap<usize, QueueItem>,
    batch_status: BatchStatus,
    detokenize_sender: mpsc::UnboundedSender<DetokenizeTask>,
    detokenize_receiver: mpsc::UnboundedReceiver<DetokenizeResult>,
    guided_decode_sender: mpsc::UnboundedSender<GuidedDecodeTask>,
    guided_decode_receiver: mpsc::UnboundedReceiver<GuidedDecodeResult>,
    perf_history: QueuePerfHistory,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueueOutputCandidate {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueueOutputToken {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<QueueOutputCandidate>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct QueueOutput {
    pub text: String,
    pub tokens: Vec<QueueOutputToken>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueFinishReason {
    Stop,
    Length,
}

impl QueueFinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueFinishMeta {
    pub reason: QueueFinishReason,
    pub matched_stop_suffix: Option<String>,
    pub matched_stop_suffix_index: Option<usize>,
    pub generated_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum QueueEvent {
    Delta(QueueOutput),
    Done(QueueFinishMeta),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchStatus {
    PrefillWithoutOutput,
    Prefill,
    Decode,
}

impl Queue {
    pub fn new(
        model_forward: Box<dyn ModelForward>,
        tokenizer: Arc<Tokenizer>,
        max_batch_size: usize,
        paragraph_len: usize,
        perf_history: QueuePerfHistory,
    ) -> Self {
        debug_assert!(max_batch_size > 0);
        debug_assert!(paragraph_len > 0);

        let guided_vocab_size = tokenizer.vocab_size();
        let guided_tokenizer_info = build_tokenizer_info_from_vocab(
            tokenizer.vocab_tokens(),
            guided_vocab_size,
            &[END_TOKEN_ID],
        );
        let (detokenize_sender, detokenize_receiver) =
            spawn_detokenize_worker(Arc::clone(&tokenizer));
        let guided_token_mask_state =
            GuidedTokenMaskBatchState::new(max_batch_size, guided_vocab_size);
        let (guided_decode_sender, guided_decode_receiver) =
            spawn_guided_decode_worker(guided_vocab_size);

        Self {
            model_forward,
            max_batch_size,
            paragraph_len,
            guided_vocab_size,
            guided_tokenizer_info,
            guided_token_mask_state,
            items: IndexMap::new(),
            batch_status: BatchStatus::PrefillWithoutOutput,
            detokenize_sender,
            detokenize_receiver,
            guided_decode_sender,
            guided_decode_receiver,
            perf_history,
        }
    }

    pub fn push(&mut self, item_id: usize, mut item: QueueItem) -> Result<(), String> {
        // 约定: item 入队前已经完成 tokenize. 首部已经被tokenize补一个 0.
        // 1. 若总长度仍不能被 paragraph_len 整除, 继续在首部补 0 直到整除.
        // 2. Waiting / Prefill 阶段, context_tokens_for_step 存完整 prompt.
        // 3. Decode 阶段, context_tokens_for_step 只存下一次 decode 要喂的 token.
        if item.context_tokens_for_step.is_empty() {
            return Err("context_tokens_for_step cannot be empty".to_string());
        }
        if item.context_tokens_for_step.len() % self.paragraph_len != 0 {
            return Err(format!(
                "context_tokens_for_step length {} must be divisible by paragraph_len {}",
                item.context_tokens_for_step.len(),
                self.paragraph_len
            ));
        }
        if self.items.contains_key(&item_id) {
            return Err(format!("duplicate item_id {item_id}"));
        }

        item.num_paragraphs = item.context_tokens_for_step.len() / self.paragraph_len;
        self.prepare_item_for_push(item_id, &mut item)?;
        self.items.insert(item_id, item);
        self.update_batch_status();
        Ok(())
    }

    pub fn step(&mut self, item_ids: &[usize]) -> Option<Vec<TokenId>> {
        if item_ids.is_empty() {
            return None;
        }

        self.assign_batch_ids(item_ids);
        self.materialize_guided_prepared_states(item_ids);
        let step_inputs = self.build_step_inputs(item_ids);
        let contexts: Vec<&[i32]> = step_inputs.contexts.iter().map(Vec::as_slice).collect();
        let context_masks: Vec<&[u8]> = step_inputs
            .context_masks
            .iter()
            .map(Vec::as_slice)
            .collect();
        let guided_token_mask_ref = step_inputs
            .has_masked_guided_token
            .then(|| self.guided_token_mask_state.as_ref());
        let step_mode =
            Self::build_step_mode(self.batch_status, &step_inputs, guided_token_mask_ref);

        self.model_forward
            .step(&step_inputs.batch_ids, &contexts, &context_masks, step_mode)
    }

    pub fn remove(&mut self, item_ids: &[usize]) {
        for item_id in item_ids {
            let Some(item) = self.items.shift_remove(item_id) else {
                continue;
            };

            if let Some(batch_id) = item.batch_id {
                self.model_forward.reset(batch_id);
                self.guided_token_mask_state.reset(batch_id);
            }
        }

        self.update_batch_status();
    }

    pub(crate) fn run_once(&mut self) -> usize {
        let items_before = self.items.len();
        if items_before == 0 {
            return 0;
        }

        self.drain_async_results();
        if self.items.is_empty() {
            self.update_batch_status();
            return items_before;
        }

        if self
            .items
            .values()
            .all(|item| item.status == QueueItemStatus::Finished)
        {
            thread::sleep(Duration::from_millis(1));
            return items_before.saturating_sub(self.items.len());
        }

        let item_ids = self.collect_step_item_ids();
        if item_ids.is_empty() {
            if self.has_pending_async_work() {
                thread::sleep(Duration::from_millis(1));
            }
            self.update_batch_status();
            return items_before.saturating_sub(self.items.len());
        }

        let batch_status = self.batch_status;
        let started_at = Instant::now();
        let step_result = self.step(&item_ids);
        self.record_perf_sample(batch_status, item_ids.len(), started_at.elapsed());

        match step_result {
            None => self.advance_prefill_items(&item_ids),
            Some(new_tokens) => self.apply_output_tokens(&item_ids, new_tokens),
        }

        self.drain_async_results();
        self.update_batch_status();
        items_before.saturating_sub(self.items.len())
    }

    pub fn run(&mut self) {
        while !self.items.is_empty() {
            self.run_once();
        }
    }

    fn drain_async_results(&mut self) {
        self.drain_detokenize_results();
        self.drain_guided_decode_results();
    }

    fn has_pending_async_work(&self) -> bool {
        self.items.values().any(|item| {
            item.pending_detokenize_tasks > 0
                || matches!(item.guided_decoding_status, GuidedDecodingStatus::Pending)
        })
    }

    fn record_perf_sample(&self, batch_status: BatchStatus, batch_used: usize, duration: Duration) {
        let stage = match batch_status {
            BatchStatus::Decode => QueuePerfStage::Decode,
            BatchStatus::PrefillWithoutOutput | BatchStatus::Prefill => QueuePerfStage::Prefill,
        };

        record_perf_sample(
            &self.perf_history,
            stage,
            batch_used,
            self.max_batch_size,
            self.paragraph_len,
            duration,
        );
    }
}

pub struct QueueItem {
    batch_id: Option<usize>,
    num_paragraphs: usize,
    context_tokens_for_step: Vec<i32>,

    sampling_config: SamplingConfig,
    token_logprobs_config: Option<TokenIdLogprobsConfig>,
    guided_decoding_config: Option<GuidedDecodingConfig>,
    guided_decoding_status: GuidedDecodingStatus,
    status: QueueItemStatus,

    tokens_in_buffer: Vec<QueueOutputToken>,
    stop_suffixes: Vec<(String, Vec<u8>)>,
    max_stop_suffix_len: usize,
    pending_stop_tokens: Vec<QueueOutputToken>,
    completions_tx: mpsc::Sender<QueueEvent>,
    pending_detokenize_tasks: usize,
    finish_meta: Option<QueueFinishMeta>,
}

impl QueueItem {
    pub fn new(
        context_tokens_for_step: Vec<i32>,
        sampling_config: SamplingConfig,
        token_logprobs_config: Option<TokenIdLogprobsConfig>,
        stop_suffixes: Vec<String>,
        completions_tx: mpsc::Sender<QueueEvent>,
        guided_decoding_config: Option<GuidedDecodingConfig>,
    ) -> Self {
        let stop_suffixes: Vec<_> = stop_suffixes
            .into_iter()
            .filter(|suffix| !suffix.is_empty())
            .map(|text| {
                let bytes = text.as_bytes().to_vec();
                (text, bytes)
            })
            .collect();
        let max_stop_suffix_len = stop_suffixes
            .iter()
            .map(|(_, bytes)| bytes.len())
            .max()
            .unwrap_or(0);
        Self {
            batch_id: None,
            num_paragraphs: 0,
            context_tokens_for_step,
            sampling_config,
            token_logprobs_config,
            guided_decoding_config,
            guided_decoding_status: GuidedDecodingStatus::Disabled,
            status: QueueItemStatus::Waiting,
            tokens_in_buffer: Vec::new(),
            stop_suffixes,
            max_stop_suffix_len,
            pending_stop_tokens: Vec::new(),
            completions_tx,
            pending_detokenize_tasks: 0,
            finish_meta: None,
        }
    }

    fn generated_tokens(&self) -> usize {
        match self.status {
            QueueItemStatus::Waiting | QueueItemStatus::Prefill(_) => 0,
            QueueItemStatus::Decode(new_tokens_len) => new_tokens_len,
            QueueItemStatus::Finished => self
                .finish_meta
                .as_ref()
                .map(|meta| meta.generated_tokens)
                .unwrap_or(0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueItemStatus {
    Waiting,
    Prefill(usize), // next_paragraph_id, 0-based
    Decode(usize),  // new_tokens_len
    Finished,
}
