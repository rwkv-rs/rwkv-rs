use std::collections::{HashMap, HashSet};
use std::iter::repeat_n;
use std::time::Instant;

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};

use super::batch_scheduler::DefaultScheduler;
use super::request_state::{ActiveRequest as InferEntry, ActiveRequestState as InferEntryState};
use super::request_submit::{InferenceSubmitCommand, InferenceSubmitHandle, InferenceSubmitResult};
use super::{
    EngineEvent, EntryId, FinishMetadata, FinishReason, OutputToken, OutputTokenCandidate,
    SampledToken, SamplingConfig, StreamDelta, TimingBreakdownMs, TokenLogprobsConfig,
};

#[derive(Clone, Debug)]
pub struct InferenceExecutionConfig {
    pub tokenizer_vocab_path: String,
    pub max_batch_size: usize,
    pub paragraph_len: usize,
    pub max_context_len: usize,
    pub decode_first: bool,
}

impl Default for InferenceExecutionConfig {
    fn default() -> Self {
        Self {
            tokenizer_vocab_path: String::new(),
            max_batch_size: 4,
            paragraph_len: 256,
            max_context_len: 4096,
            decode_first: true,
        }
    }
}

/// Model forward inference interface. Prefill and decode are unified through this interface.
pub trait ModelForward: Send + 'static {
    /// Forward inference + sampling.
    /// - `batch_ids` / `contexts` / `masks`: parallel slices with identical length.
    ///   Each index represents one active batch position:
    ///   - prefill chunk: `contexts[i].len() == paragraph_len`
    ///   - decode step: `contexts[i].len() == 1`
    /// - `samplings`: per-position sampling parameters (one per active position)
    /// - `token_logprobs`: optional per-position logprob output configuration
    /// - `need_sample`: when false, skip unembed+sampling (prefill intermediate chunk), return empty Vec
    /// Returns sampled tokens with optional logprob metadata built from post-sampling probabilities.
    fn forward(
        &mut self,
        batch_ids: &[usize],
        contexts: &[&[i32]],
        masks: &[&[u8]],
        samplings: &[SamplingConfig],
        token_logprobs: &[Option<TokenLogprobsConfig>],
        need_sample: bool,
    ) -> crate::Result<Vec<SampledToken>>;

    /// Reset recurrent state for the given batch position.
    fn reset(&mut self, batch_index: usize) -> crate::Result<()>;
}

struct PendingSubmit {
    entry_id: EntryId,
    input_text: String,
    sampling: SamplingConfig,
    stop_suffixes: Vec<String>,
    token_logprobs: Option<TokenLogprobsConfig>,
    stream: bool,
    submitted_at: Instant,
    runtime_received_at: Instant,
    validate_ms: Option<u64>,
    reply: oneshot::Sender<InferenceSubmitResult>,
}

#[derive(Default)]
struct ForwardBatch {
    entry_ids: Vec<EntryId>,
    batch_ids: Vec<usize>,
    contexts: Vec<Vec<i32>>,
    context_masks: Vec<Vec<u8>>,
    sampling_configs: Vec<SamplingConfig>,
    token_logprob_configs: Vec<Option<TokenLogprobsConfig>>,
    batch_to_entry: HashMap<usize, EntryId>,
}

impl ForwardBatch {
    fn push(
        &mut self,
        entry_id: EntryId,
        batch_index: usize,
        context: Vec<i32>,
        context_mask: Vec<u8>,
        sampling: SamplingConfig,
        token_logprobs: Option<TokenLogprobsConfig>,
    ) {
        self.entry_ids.push(entry_id);
        self.batch_ids.push(batch_index);
        self.contexts.push(context);
        self.context_masks.push(context_mask);
        self.sampling_configs.push(sampling);
        self.token_logprob_configs.push(token_logprobs);
        self.batch_to_entry.insert(batch_index, entry_id);
    }

    fn is_empty(&self) -> bool {
        self.batch_ids.is_empty()
    }

    fn entry_ids(&self) -> &[EntryId] {
        &self.entry_ids
    }
}

#[derive(Clone, Copy, Debug)]
enum SampleSource {
    Decode,
    PrefillLastChunk,
}

struct SampleApplyOutcome {
    entry_id: EntryId,
    stream_tx: Option<mpsc::Sender<EngineEvent>>,
    delta: Option<StreamDelta>,
    tail: Option<StreamDelta>,
    finish_meta: Option<FinishMetadata>,
    finished_batch_index: Option<usize>,
}

pub struct InferenceExecutionLoop {
    cfg: EngineRuntimeConfig,
    tokenizer: Tokenizer,
    scheduler: DefaultScheduler,
    entries: HashMap<EntryId, InferEntry>,
    rx: mpsc::Receiver<InferenceSubmitCommand>,
    executor: Box<dyn ModelForward>,
}

impl InferenceExecutionLoop {
    pub fn spawn(
        cfg: EngineRuntimeConfig,
        executor: Box<dyn ModelForward>,
    ) -> crate::Result<InferenceSubmitHandle> {
        let tokenizer = Tokenizer::new(&cfg.tokenizer_vocab_path).map_err(|e| {
            crate::Error::bad_request(format!(
                "failed to load tokenizer vocab {}: {e}",
                cfg.tokenizer_vocab_path
            ))
        })?;

        let (tx, rx) = mpsc::channel(1024);
        let handle = InferenceSubmitHandle::new(tx);
        let rt = Self::new(cfg, tokenizer, rx, executor);
        tokio::spawn(async move {
            rt.run().await;
        });
        Ok(handle)
    }

    pub fn new(
        cfg: EngineRuntimeConfig,
        tokenizer: Tokenizer,
        rx: mpsc::Receiver<InferenceSubmitCommand>,
        executor: Box<dyn ModelForward>,
    ) -> Self {
        let scheduler = DefaultScheduler::new(cfg.max_batch_size);
        Self {
            cfg,
            tokenizer,
            scheduler,
            entries: HashMap::new(),
            rx,
            executor,
        }
    }

    pub async fn run(mut self) {
        loop {
            self.drain_pending_commands().await;

            if !self.has_ready_work() && !self.wait_for_commands().await {
                break;
            }

            while self.has_ready_work() {
                self.drain_pending_commands().await;

                if !self.has_ready_work() {
                    break;
                }

                if !self.tick_once().await {
                    break;
                }
            }
        }
    }

    fn handle_command(
        &mut self,
        cmd: InferenceSubmitCommand,
        pending_submits: &mut Vec<PendingSubmit>,
        cancelled_entry_ids: &mut Vec<EntryId>,
    ) {
        match cmd {
            InferenceSubmitCommand::SubmitText {
                entry_id,
                input_text,
                sampling,
                stop_suffixes,
                token_logprobs,
                stream,
                submitted_at,
                validate_ms,
                reply,
            } => pending_submits.push(PendingSubmit {
                entry_id,
                input_text,
                sampling,
                stop_suffixes,
                token_logprobs,
                stream,
                submitted_at,
                runtime_received_at: Instant::now(),
                validate_ms,
                reply,
            }),
            InferenceSubmitCommand::Cancel { entry_id } => {
                cancelled_entry_ids.push(entry_id);
            }
        }
    }

    async fn finish_command_batch(
        &mut self,
        pending_submits: Vec<PendingSubmit>,
        cancelled_entry_ids: Vec<EntryId>,
    ) {
        if !cancelled_entry_ids.is_empty() {
            self.cancel_entries(&cancelled_entry_ids).await;
        }

        #[cfg(feature = "trace")]
        if !pending_submits.is_empty() {
            tracing::info!(
                pending_submit = pending_submits.len(),
                active_entries = self.entries.len(),
                "runtime drained submit queue"
            );
        }

        self.handle_submit_batch(pending_submits);
    }

    async fn drain_pending_commands(&mut self) {
        let mut pending_submits = Vec::new();
        let mut cancelled_entry_ids = Vec::new();
        {
            rwkv_bench::trace_lite_scope!("rwkv.infer.engine.runtime.run");
            while let Ok(cmd) = self.rx.try_recv() {
                self.handle_command(cmd, &mut pending_submits, &mut cancelled_entry_ids);
            }
        }

        self.finish_command_batch(pending_submits, cancelled_entry_ids)
            .await;
    }

    async fn wait_for_commands(&mut self) -> bool {
        let mut pending_submits = Vec::new();
        let mut cancelled_entry_ids = Vec::new();
        {
            rwkv_bench::trace_lite_scope!("rwkv.infer.engine.runtime.run");
            let Some(cmd) = self.rx.recv().await else {
                return false;
            };
            self.handle_command(cmd, &mut pending_submits, &mut cancelled_entry_ids);

            while let Ok(cmd) = self.rx.try_recv() {
                self.handle_command(cmd, &mut pending_submits, &mut cancelled_entry_ids);
            }
        }

        self.finish_command_batch(pending_submits, cancelled_entry_ids)
            .await;
        true
    }

    fn has_ready_work(&self) -> bool {
        self.scheduler.has_ready_decode(&self.entries)
            || self.scheduler.has_ready_prefill(&self.entries)
            || (self.scheduler.has_free_slot() && self.scheduler.has_waiting_entries(&self.entries))
    }

    fn handle_submit_batch(&mut self, submits: Vec<PendingSubmit>) {
        rwkv_bench::trace_scope!("rwkv.infer.engine.runtime.submit");
        if submits.is_empty() {
            return;
        }

        let tokenize_start = Instant::now();
        let texts: Vec<String> = submits
            .iter()
            .map(|submit| submit.input_text.clone())
            .collect();
        let tokenized: Vec<Vec<u16>> = self.tokenizer.encode_batch(texts, false);
        let tokenize_ms = tokenize_start.elapsed().as_millis() as u64;

        #[cfg(feature = "trace")]
        tracing::info!(
            stage = "tokenize",
            batch_size = tokenized.len(),
            tokenize_ms,
            "submit batch tokenized"
        );

        for (submit, token_ids_u16) in submits.into_iter().zip(tokenized.into_iter()) {
            #[cfg(feature = "trace")]
            let submit_span = tracing::info_span!(
                "rwkv.infer.request",
                request_id = %submit.entry_id,
                stream = submit.stream
            );
            #[cfg(feature = "trace")]
            let _submit_guard = submit_span.enter();

            let token_ids: Vec<i32> = token_ids_u16.into_iter().map(|t| t as i32).collect();
            if token_ids.len() > self.cfg.max_context_len {
                let _ = submit.reply.send(InferenceSubmitResult::Error {
                    entry_id: submit.entry_id,
                    message: format!(
                        "prefill too long: {} tokens > max_context_len={}",
                        token_ids.len(),
                        self.cfg.max_context_len
                    ),
                });
                continue;
            }

            let mut entry = InferEntry::new(
                submit.entry_id,
                submit.input_text,
                submit.sampling,
                submit
                    .stop_suffixes
                    .into_iter()
                    .filter(|suffix| !suffix.is_empty())
                    .collect(),
                submit.token_logprobs,
            );
            entry.input_token_ids = token_ids;
            entry.validate_ms = submit.validate_ms;
            entry.tokenize_ms = Some(tokenize_ms);
            entry.submitted_at = Some(submit.submitted_at);
            entry.runtime_received_at = Some(submit.runtime_received_at);
            entry.enqueued_at = Some(Instant::now());
            #[cfg(feature = "trace")]
            tracing::info!(
                request_id = %submit.entry_id,
                prompt_tokens = entry.input_token_ids.len(),
                stream = submit.stream,
                "request enqueued"
            );

            if submit.stream {
                let (tx, rx) = mpsc::channel(256);
                entry.stream_tx = Some(tx);
                self.entries.insert(submit.entry_id, entry);
                self.scheduler.push_waiting(submit.entry_id);
                let _ = submit.reply.send(InferenceSubmitResult::Stream {
                    entry_id: submit.entry_id,
                    rx,
                });
            } else {
                self.entries.insert(submit.entry_id, entry);
                self.scheduler.push_waiting(submit.entry_id);
                let _ = submit.reply.send(InferenceSubmitResult::Done {
                    entry_id: submit.entry_id,
                    output_text: String::new(),
                    token_ids: Vec::new(),
                });
            }
        }
    }

    async fn cancel_entries(&mut self, entry_ids: &[EntryId]) {
        for entry_id in entry_ids.iter().copied() {
            self.terminate_entry_with_error(
                entry_id,
                InferEntryState::Cancelled,
                "request cancelled".to_string(),
            )
            .await;
        }
    }

    async fn terminate_entry_with_error(
        &mut self,
        entry_id: EntryId,
        state: InferEntryState,
        message: String,
    ) {
        let (stream_tx, batch_index) = match self.entries.get_mut(&entry_id) {
            Some(entry) => {
                entry.state = state;
                (entry.stream_tx.clone(), entry.batch_index.take())
            }
            None => return,
        };

        if let Some(tx) = stream_tx {
            let _ = tx.send(EngineEvent::Error(message.clone())).await;
        }
        if let Some(batch_index) = batch_index {
            let _ = self.executor.reset(batch_index);
        }
        self.scheduler.on_done(entry_id);
        self.entries.remove(&entry_id);
    }

    async fn tick_once(&mut self) -> bool {
        let step = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.engine.runtime.tick");
            self.scheduler.schedule(&mut self.entries)
        };

        if !step.has_work() {
            return false;
        }

        if self.cfg.decode_first {
            self.run_decode(&step.decode_ids).await;
            self.run_prefill(&step.prefill_ids).await;
        } else {
            self.run_prefill(&step.prefill_ids).await;
            self.run_decode(&step.decode_ids).await;
        }

        true
    }

    fn has_stream_delta_output(delta: &StreamDelta) -> bool {
        !delta.text.is_empty() || !delta.tokens.is_empty()
    }

    async fn run_prefill(&mut self, prefill_ids: &[EntryId]) {
        if prefill_ids.is_empty() {
            return;
        }

        let mut prefill_batch = ForwardBatch::default();
        let mut sample_prefill_batch = ForwardBatch::default();

        {
            rwkv_bench::trace_scope!("rwkv.infer.engine.runtime.prefill");
            let paragraph_len = self.cfg.paragraph_len;
            for entry_id in prefill_ids.iter().copied() {
                let Some(entry) = self.entries.get_mut(&entry_id) else {
                    continue;
                };
                let Some(batch_index) = entry.batch_index else {
                    continue;
                };

                let chunk_issued_at = Instant::now();
                if entry.first_prefill_at.is_none() {
                    entry.first_prefill_at = Some(chunk_issued_at);
                }

                if entry.prefill_padded_token_ids.is_empty() {
                    let prefill_len = entry.input_token_ids.len();
                    let pad_len = (paragraph_len - (prefill_len % paragraph_len)) % paragraph_len;
                    entry.prefill_pad_len = pad_len;
                    entry.prefill_padded_token_ids = repeat_n(0, pad_len)
                        .chain(entry.input_token_ids.iter().copied())
                        .collect();
                    entry.prefill_chunk_cursor = 0;
                }

                let padded_len = entry.prefill_padded_token_ids.len();
                let start = entry.prefill_chunk_cursor * paragraph_len;
                if start >= padded_len {
                    entry.state = InferEntryState::RunningDecode;
                    continue;
                }
                let end = (start + paragraph_len).min(padded_len);
                let is_last_chunk = end == padded_len;

                #[cfg(feature = "trace")]
                let _prefill_chunk_span = tracing::info_span!(
                    "rwkv.infer.prefill.chunk",
                    request_id = %entry_id,
                    batch_index,
                    chunk_idx = entry.prefill_chunk_cursor,
                    chunk_tokens = end.saturating_sub(start),
                    paragraph_len,
                    needs_sample = is_last_chunk,
                )
                .entered();

                let token_ids = entry.prefill_padded_token_ids[start..end].to_vec();
                let mut context_mask = vec![1u8; token_ids.len()];
                if entry.prefill_chunk_cursor == 0 && entry.prefill_pad_len > 0 {
                    for index in 0..entry.prefill_pad_len.min(context_mask.len()) {
                        context_mask[index] = 0;
                    }
                }

                entry.prefill_chunk_cursor += 1;
                entry.last_prefill_at = Some(chunk_issued_at);

                if is_last_chunk {
                    entry.prefill_padded_token_ids.clear();
                    entry.prefill_pad_len = 0;
                    entry.prefill_chunk_cursor = 0;
                    sample_prefill_batch.push(
                        entry_id,
                        batch_index,
                        token_ids,
                        context_mask,
                        entry.sampling,
                        entry.token_logprobs.clone(),
                    );
                } else {
                    prefill_batch.push(
                        entry_id,
                        batch_index,
                        token_ids,
                        context_mask,
                        entry.sampling,
                        None,
                    );
                }
            }
        }

        if !prefill_batch.is_empty() {
            #[cfg(feature = "trace")]
            tracing::info!(
                stage = "prefill_step",
                batch_size = prefill_batch.batch_ids.len(),
                "dispatch prefill batch"
            );

            let prefill_result = {
                let contexts_ref: Vec<&[i32]> = prefill_batch
                    .contexts
                    .iter()
                    .map(|ctx| ctx.as_slice())
                    .collect();
                let masks_ref: Vec<&[u8]> = prefill_batch
                    .context_masks
                    .iter()
                    .map(|mask| mask.as_slice())
                    .collect();
                #[cfg(feature = "nsys")]
                let _nvtx_prefill = nvtx::range!("rwkv.infer.prefill");
                self.executor.forward(
                    &prefill_batch.batch_ids,
                    &contexts_ref,
                    &masks_ref,
                    &prefill_batch.sampling_configs,
                    &prefill_batch.token_logprob_configs,
                    false,
                )
            };

            if let Err(err) = prefill_result {
                let chain = err.format_chain();
                log::error!("prefill failed: {chain}");
                self.fail_entries(
                    prefill_batch.entry_ids(),
                    format!("prefill failed: {chain}"),
                )
                .await;
            }
        }

        if sample_prefill_batch.is_empty() {
            return;
        }

        #[cfg(feature = "trace")]
        tracing::info!(
            stage = "prefill_sample_step",
            batch_size = sample_prefill_batch.batch_ids.len(),
            "dispatch sampled prefill batch"
        );

        let sample_prefill_result = {
            let contexts_ref: Vec<&[i32]> = sample_prefill_batch
                .contexts
                .iter()
                .map(|ctx| ctx.as_slice())
                .collect();
            let masks_ref: Vec<&[u8]> = sample_prefill_batch
                .context_masks
                .iter()
                .map(|mask| mask.as_slice())
                .collect();
            #[cfg(feature = "nsys")]
            let _nvtx_prefill = nvtx::range!("rwkv.infer.prefill.sample");
            self.executor.forward(
                &sample_prefill_batch.batch_ids,
                &contexts_ref,
                &masks_ref,
                &sample_prefill_batch.sampling_configs,
                &sample_prefill_batch.token_logprob_configs,
                true,
            )
        };

        match sample_prefill_result {
            Ok(sampled_tokens) => {
                self.apply_sampled_tokens(
                    sampled_tokens,
                    &sample_prefill_batch.batch_to_entry,
                    SampleSource::PrefillLastChunk,
                )
                .await;
            }
            Err(err) => {
                let chain = err.format_chain();
                log::error!("prefill sample failed: {chain}");
                self.fail_entries(
                    sample_prefill_batch.entry_ids(),
                    format!("prefill sample failed: {chain}"),
                )
                .await;
            }
        }
    }

    async fn run_decode(&mut self, decode_ids: &[EntryId]) {
        if decode_ids.is_empty() {
            return;
        }

        let mut decode_batch = ForwardBatch::default();
        for entry_id in decode_ids.iter().copied() {
            let Some(entry) = self.entries.get_mut(&entry_id) else {
                continue;
            };
            let Some(batch_index) = entry.batch_index else {
                continue;
            };

            let last_token_id = entry
                .generated_token_ids
                .last()
                .copied()
                .or_else(|| entry.input_token_ids.last().copied())
                .unwrap_or(0);

            decode_batch.push(
                entry_id,
                batch_index,
                vec![last_token_id],
                vec![1u8],
                entry.sampling,
                entry.token_logprobs.clone(),
            );
        }

        if decode_batch.is_empty() {
            return;
        }

        #[cfg(feature = "trace")]
        tracing::info!(
            target: "rwkv.infer",
            stage = "decode_step",
            batch_size = decode_batch.batch_ids.len(),
            "dispatch decode batch"
        );

        let decode_result = {
            rwkv_bench::trace_scope!("rwkv.infer.engine.runtime.decode");
            #[cfg(feature = "nsys")]
            let _nvtx_decode = nvtx::range!("rwkv.infer.decode");
            let contexts_ref: Vec<&[i32]> = decode_batch
                .contexts
                .iter()
                .map(|context| context.as_slice())
                .collect();
            let masks_ref: Vec<&[u8]> = decode_batch
                .context_masks
                .iter()
                .map(|mask| mask.as_slice())
                .collect();
            self.executor.forward(
                &decode_batch.batch_ids,
                &contexts_ref,
                &masks_ref,
                &decode_batch.sampling_configs,
                &decode_batch.token_logprob_configs,
                true,
            )
        };
        match decode_result {
            Ok(sampled_tokens) => {
                self.apply_sampled_tokens(
                    sampled_tokens,
                    &decode_batch.batch_to_entry,
                    SampleSource::Decode,
                )
                .await;
            }
            Err(err) => {
                let chain = err.format_chain();
                log::error!("decode failed: {chain}");
                self.fail_entries(decode_batch.entry_ids(), format!("decode failed: {chain}"))
                    .await;
            }
        }
    }

    async fn apply_sampled_tokens(
        &mut self,
        sampled_tokens: Vec<SampledToken>,
        batch_to_entry: &HashMap<usize, EntryId>,
        source: SampleSource,
    ) {
        let mut seen_batch_indices = HashSet::new();
        for sampled_token in sampled_tokens {
            let batch_index = sampled_token.batch_index;
            let Some(entry_id) = batch_to_entry.get(&batch_index).copied() else {
                continue;
            };
            seen_batch_indices.insert(batch_index);

            let Some(outcome) = self.prepare_sampled_token_outcome(entry_id, sampled_token, source)
            else {
                continue;
            };
            self.emit_sampled_token_outcome(outcome).await;
        }

        let missing_entry_ids: Vec<EntryId> = batch_to_entry
            .iter()
            .filter_map(|(batch_index, entry_id)| {
                (!seen_batch_indices.contains(batch_index)).then_some(*entry_id)
            })
            .collect();
        if !missing_entry_ids.is_empty() {
            self.fail_entries(
                &missing_entry_ids,
                "executor returned incomplete sample output".to_string(),
            )
            .await;
        }
    }

    fn prepare_sampled_token_outcome(
        &mut self,
        entry_id: EntryId,
        sampled_token: SampledToken,
        source: SampleSource,
    ) -> Option<SampleApplyOutcome> {
        let entry = self.entries.get_mut(&entry_id)?;

        if matches!(source, SampleSource::Decode) {
            let decode_at = Instant::now();
            if entry.first_decode_at.is_none() {
                entry.first_decode_at = Some(decode_at);
            }
            entry.last_decode_at = Some(decode_at);

            #[cfg(feature = "trace")]
            tracing::trace!(
                target: "rwkv.infer",
                request_id = %entry.entry_id,
                batch_index = sampled_token.batch_index,
                generated_tokens = entry.generated_token_ids.len(),
                "decode token"
            );
        }

        let output_token = sampled_token_to_output(&self.tokenizer, &sampled_token);
        entry.push_generated_token(output_token, sampled_token.token_id);

        let (delta, matched_stop) = entry.emit_stream_delta();
        let delta = if Self::has_stream_delta_output(&delta) {
            if entry.first_emit_at.is_none() {
                entry.first_emit_at = Some(Instant::now());
            }
            Some(delta)
        } else {
            None
        };

        let matched_stop_index = matched_stop.map(|stop| stop.index);
        let hit_stop_suffix = matched_stop_index.is_some();
        let hit_max_new_tokens = entry.generated_token_ids.len() >= entry.max_new_tokens;
        let finish_meta = finish_metadata_for_entry(
            entry,
            matched_stop_index,
            hit_max_new_tokens,
            Instant::now(),
        );

        let stream_tx = entry.stream_tx.clone();
        let mut tail = None;
        let mut finished_batch_index = None;
        if finish_meta.is_some() {
            let next_tail = if hit_stop_suffix {
                entry.flush_stream_delta_until(entry.stop_trunc_len(), true)
            } else {
                entry.flush_stream_delta_until(entry.generated_bytes.len(), true)
            };
            if Self::has_stream_delta_output(&next_tail) {
                if entry.first_emit_at.is_none() {
                    entry.first_emit_at = Some(Instant::now());
                }
                tail = Some(next_tail);
            }
            entry.state = InferEntryState::Done;
            finished_batch_index = entry.batch_index.take();
        } else if matches!(source, SampleSource::PrefillLastChunk) {
            entry.state = InferEntryState::RunningDecode;
        }

        Some(SampleApplyOutcome {
            entry_id,
            stream_tx,
            delta,
            tail,
            finish_meta,
            finished_batch_index,
        })
    }

    async fn emit_sampled_token_outcome(&mut self, outcome: SampleApplyOutcome) {
        let SampleApplyOutcome {
            entry_id,
            stream_tx,
            delta,
            tail,
            finish_meta,
            finished_batch_index,
        } = outcome;

        if let Some(tx) = stream_tx {
            if let Some(delta) = delta {
                #[cfg(feature = "trace")]
                tracing::trace!(
                    request_id = %entry_id,
                    chars = delta.text.chars().count(),
                    tokens = delta.tokens.len(),
                    "stream emit chunk"
                );
                let _ = tx.send(EngineEvent::Output(delta)).await;
            }

            if let Some(meta) = finish_meta.as_ref() {
                if let Some(tail) = tail {
                    let _ = tx.send(EngineEvent::Output(tail)).await;
                }
                let _ = tx.send(EngineEvent::Done(meta.clone())).await;
            }
        }

        if let Some(meta) = finish_meta {
            #[cfg(feature = "trace")]
            tracing::info!(
                request_id = %entry_id,
                finish_reason = meta.reason.as_openai_str(),
                generated_tokens = meta.generated_tokens,
                max_new_tokens = meta.max_new_tokens,
                queue_wait_ms = meta.timings_ms.as_ref().and_then(|t| t.queue_wait_ms),
                schedule_wait_ms = meta.timings_ms.as_ref().and_then(|t| t.schedule_wait_ms),
                prefill_first_ms = meta.timings_ms.as_ref().and_then(|t| t.prefill_first_ms),
                first_emit_ms = meta.timings_ms.as_ref().and_then(|t| t.first_emit_ms),
                prefill_total_ms = meta.timings_ms.as_ref().and_then(|t| t.prefill_total_ms),
                decode_total_ms = meta.timings_ms.as_ref().and_then(|t| t.decode_total_ms),
                request_total_ms = meta.timings_ms.as_ref().and_then(|t| t.request_total_ms),
                "request finished"
            );
            match meta.reason {
                FinishReason::Stop => {
                    log::info!(
                        "generation finished: finish_reason=stop matched_stop_suffix_index={} matched_stop_suffix={:?} generated_tokens={} max_tokens={}",
                        meta.matched_stop_suffix_index.unwrap_or_default(),
                        meta.matched_stop_suffix.as_deref().unwrap_or(""),
                        meta.generated_tokens,
                        meta.max_new_tokens
                    );
                }
                FinishReason::Length => {
                    log::info!(
                        "generation finished: finish_reason=length generated_tokens={} max_tokens={}",
                        meta.generated_tokens,
                        meta.max_new_tokens
                    );
                }
            }
            if let Some(batch_index) = finished_batch_index {
                let _ = self.executor.reset(batch_index);
            }
            self.scheduler.on_done(entry_id);
            self.entries.remove(&entry_id);
        }
    }

    async fn fail_entries(&mut self, entry_ids: &[EntryId], message: String) {
        for entry_id in entry_ids.iter().copied() {
            #[cfg(feature = "trace")]
            tracing::error!(request_id = %entry_id, error = %message, "entry failed");
            self.terminate_entry_with_error(entry_id, InferEntryState::Failed, message.clone())
                .await;
        }
    }
}

fn sampled_token_to_output(tokenizer: &Tokenizer, sampled_token: &SampledToken) -> OutputToken {
    let token_bytes = tokenizer
        .token_bytes(sampled_token.token_id as u16)
        .to_vec();
    let token = String::from_utf8_lossy(&token_bytes).into_owned();
    let (logprob, top_logprobs) = match sampled_token.logprob.as_ref() {
        Some(logprob) => (
            Some(logprob.logprob),
            logprob
                .top_logprobs
                .iter()
                .map(|candidate| {
                    let bytes = tokenizer.token_bytes(candidate.token_id as u16).to_vec();
                    OutputTokenCandidate {
                        token: String::from_utf8_lossy(&bytes).into_owned(),
                        bytes,
                        logprob: candidate.logprob,
                    }
                })
                .collect(),
        ),
        None => (None, Vec::new()),
    };

    OutputToken {
        token,
        bytes: token_bytes,
        logprob,
        top_logprobs,
    }
}

fn finish_metadata_for_entry(
    entry: &InferEntry,
    matched_stop_index: Option<usize>,
    hit_max_new_tokens: bool,
    finished_at: Instant,
) -> Option<FinishMetadata> {
    let timings_ms = build_timing_breakdown(entry, finished_at);

    if let Some(index) = matched_stop_index {
        let matched_suffix = entry
            .stop_suffixes
            .get(index)
            .map(|suffix| String::from_utf8_lossy(suffix).into_owned())
            .unwrap_or_default();
        return Some(FinishMetadata {
            reason: FinishReason::Stop,
            matched_stop_suffix: Some(matched_suffix),
            matched_stop_suffix_index: Some(index),
            max_new_tokens: entry.max_new_tokens,
            generated_tokens: entry.generated_token_ids.len(),
            timings_ms,
        });
    }

    if hit_max_new_tokens {
        return Some(FinishMetadata {
            reason: FinishReason::Length,
            matched_stop_suffix: None,
            matched_stop_suffix_index: None,
            max_new_tokens: entry.max_new_tokens,
            generated_tokens: entry.generated_token_ids.len(),
            timings_ms,
        });
    }

    None
}

fn build_timing_breakdown(entry: &InferEntry, finished_at: Instant) -> Option<TimingBreakdownMs> {
    let timings = TimingBreakdownMs {
        validate_ms: entry.validate_ms,
        tokenize_ms: entry.tokenize_ms,
        queue_wait_ms: duration_ms_between(entry.submitted_at, entry.runtime_received_at),
        schedule_wait_ms: duration_ms_between(entry.enqueued_at, entry.scheduled_at),
        prefill_first_ms: duration_ms_between(entry.scheduled_at, entry.first_prefill_at),
        first_emit_ms: duration_ms_between(entry.first_prefill_at, entry.first_emit_at)
            .or_else(|| duration_ms_since(entry.first_prefill_at, finished_at)),
        prefill_total_ms: duration_ms_between(entry.first_prefill_at, entry.last_prefill_at)
            .or_else(|| duration_ms_between(entry.first_prefill_at, entry.first_decode_at))
            .or_else(|| duration_ms_since(entry.first_prefill_at, finished_at)),
        decode_total_ms: duration_ms_between(entry.first_decode_at, entry.last_decode_at)
            .or_else(|| duration_ms_since(entry.first_decode_at, finished_at)),
        request_total_ms: duration_ms_between(entry.submitted_at, Some(finished_at))
            .or_else(|| duration_ms_between(entry.runtime_received_at, Some(finished_at))),
    };

    if timings.validate_ms.is_some()
        || timings.tokenize_ms.is_some()
        || timings.queue_wait_ms.is_some()
        || timings.schedule_wait_ms.is_some()
        || timings.prefill_first_ms.is_some()
        || timings.first_emit_ms.is_some()
        || timings.prefill_total_ms.is_some()
        || timings.decode_total_ms.is_some()
        || timings.request_total_ms.is_some()
    {
        Some(timings)
    } else {
        None
    }
}

fn duration_ms_between(start: Option<Instant>, end: Option<Instant>) -> Option<u64> {
    let (Some(start), Some(end)) = (start, end) else {
        return None;
    };
    Some(end.saturating_duration_since(start).as_millis() as u64)
}

fn duration_ms_since(start: Option<Instant>, end: Instant) -> Option<u64> {
    let Some(start) = start else {
        return None;
    };
    Some(end.saturating_duration_since(start).as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::{EngineRuntime, EngineRuntimeConfig, ModelForward, finish_metadata_for_entry};
    use crate::inference_core::request_submit::{InferenceSubmitCommand, InferenceSubmitResult};
    use crate::inference_core::{
        EngineEvent, FinishReason, SampledToken, SamplingConfig, TokenLogprobsConfig,
    };
    use crate::inference_core::{InferEntry, InferEntryState};
    use rwkv_data::tokenizer::Tokenizer;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tokio::sync::{mpsc, oneshot};
    use tokio::time::timeout;
    use uuid::Uuid;

    const TEST_VOCAB_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/rwkv-lm/assets/rwkv_vocab_v20230424.txt"
    );

    #[derive(Clone, Default)]
    struct TestForwardState {
        forward_calls: Arc<Mutex<Vec<(bool, usize)>>>,
        reset_batch_indices: Arc<Mutex<Vec<usize>>>,
    }

    struct TestForward {
        state: TestForwardState,
        decode_token: i32,
    }

    impl TestForward {
        fn new(state: TestForwardState, decode_token: i32) -> Self {
            Self {
                state,
                decode_token,
            }
        }
    }

    impl ModelForward for TestForward {
        fn forward(
            &mut self,
            batch_ids: &[usize],
            _contexts: &[&[i32]],
            _masks: &[&[u8]],
            _samplings: &[SamplingConfig],
            _token_logprobs: &[Option<TokenLogprobsConfig>],
            need_sample: bool,
        ) -> crate::Result<Vec<SampledToken>> {
            self.state
                .forward_calls
                .lock()
                .expect("forward lock")
                .push((need_sample, batch_ids.len()));

            if need_sample {
                Ok(batch_ids
                    .iter()
                    .copied()
                    .map(|batch_index| SampledToken {
                        batch_index,
                        token_id: self.decode_token,
                        logprob: None,
                    })
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }

        fn reset(&mut self, batch_index: usize) -> crate::Result<()> {
            self.state
                .reset_batch_indices
                .lock()
                .expect("reset lock")
                .push(batch_index);
            Ok(())
        }
    }

    fn test_config() -> EngineRuntimeConfig {
        test_config_with_paragraph_len(256)
    }

    fn test_config_with_paragraph_len(paragraph_len: usize) -> EngineRuntimeConfig {
        EngineRuntimeConfig {
            tokenizer_vocab_path: TEST_VOCAB_PATH.to_string(),
            max_batch_size: 2,
            paragraph_len,
            max_context_len: 4096,
            decode_first: true,
        }
    }

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::new(TEST_VOCAB_PATH).expect("test tokenizer")
    }

    fn test_runtime() -> (EngineRuntime, TestForwardState) {
        test_runtime_with_config(test_config())
    }

    fn test_runtime_with_config(cfg: EngineRuntimeConfig) -> (EngineRuntime, TestForwardState) {
        let state = TestForwardState::default();
        let (_tx, rx) = mpsc::channel(8);
        let runtime = EngineRuntime::new(
            cfg,
            test_tokenizer(),
            rx,
            Box::new(TestForward::new(state.clone(), 1)),
        );
        (runtime, state)
    }

    fn insert_waiting_entry(
        runtime: &mut EngineRuntime,
        entry_id: Uuid,
        input_token_ids: Vec<i32>,
        max_new_tokens: usize,
        stream_tx: Option<mpsc::Sender<EngineEvent>>,
    ) {
        let mut entry = InferEntry::new(
            entry_id,
            "prompt".to_string(),
            SamplingConfig {
                max_new_tokens,
                ..SamplingConfig::default()
            },
            Vec::new(),
            None,
        );
        entry.input_token_ids = input_token_ids;
        entry.stream_tx = stream_tx;
        entry.enqueued_at = Some(Instant::now());
        runtime.entries.insert(entry_id, entry);
        runtime.scheduler.push_waiting(entry_id);
    }

    #[test]
    fn finish_metadata_prefers_stop_over_length() {
        let sampling = SamplingConfig {
            max_new_tokens: 1,
            ..SamplingConfig::default()
        };
        let mut entry = InferEntry::new(
            Uuid::new_v4(),
            "prompt".to_string(),
            sampling,
            vec!["stop".to_string()],
            None,
        );
        entry.generated_token_ids.push(1);
        let meta =
            finish_metadata_for_entry(&entry, Some(0), true, Instant::now()).expect("must finish");
        assert_eq!(meta.reason, FinishReason::Stop);
        assert_eq!(meta.matched_stop_suffix.as_deref(), Some("stop"));
        assert_eq!(meta.matched_stop_suffix_index, Some(0));
        assert_eq!(meta.generated_tokens, 1);
        assert_eq!(meta.max_new_tokens, 1);
        assert_eq!(meta.timings_ms, None);
    }

    #[test]
    fn finish_metadata_reports_length_when_no_stop_match() {
        let sampling = SamplingConfig {
            max_new_tokens: 2,
            ..SamplingConfig::default()
        };
        let mut entry = InferEntry::new(
            Uuid::new_v4(),
            "prompt".to_string(),
            sampling,
            Vec::new(),
            None,
        );
        entry.generated_token_ids.extend([1, 2]);
        let meta =
            finish_metadata_for_entry(&entry, None, true, Instant::now()).expect("must finish");
        assert_eq!(meta.reason, FinishReason::Length);
        assert_eq!(meta.matched_stop_suffix, None);
        assert_eq!(meta.matched_stop_suffix_index, None);
        assert_eq!(meta.generated_tokens, 2);
        assert_eq!(meta.max_new_tokens, 2);
        assert_eq!(meta.timings_ms, None);
    }

    #[tokio::test]
    async fn tick_once_samples_last_prefill_chunk_without_extra_decode() {
        let (mut runtime, state) = test_runtime();
        let entry_id = Uuid::new_v4();
        insert_waiting_entry(&mut runtime, entry_id, vec![11], 1, None);

        assert!(runtime.has_ready_work());
        assert!(runtime.tick_once().await);
        assert!(!runtime.entries.contains_key(&entry_id));
        assert!(!runtime.has_ready_work());
        assert_eq!(
            *state.forward_calls.lock().expect("forward calls"),
            vec![(true, 1)]
        );
        assert_eq!(
            *state.reset_batch_indices.lock().expect("reset calls"),
            vec![0]
        );
    }

    #[tokio::test]
    async fn multi_chunk_prefill_runs_nonsample_then_last_chunk_sample() {
        let (mut runtime, state) = test_runtime_with_config(test_config_with_paragraph_len(2));
        let entry_id = Uuid::new_v4();
        insert_waiting_entry(&mut runtime, entry_id, vec![10, 11, 12, 13], 1, None);

        assert!(runtime.tick_once().await);
        assert_eq!(
            runtime.entries.get(&entry_id).map(|entry| entry.state),
            Some(InferEntryState::RunningPrefill)
        );
        assert_eq!(
            *state.forward_calls.lock().expect("forward calls"),
            vec![(false, 1)]
        );

        assert!(runtime.tick_once().await);
        assert!(!runtime.entries.contains_key(&entry_id));
        assert_eq!(
            *state.forward_calls.lock().expect("forward calls"),
            vec![(false, 1), (true, 1)]
        );
        assert_eq!(
            *state.reset_batch_indices.lock().expect("reset calls"),
            vec![0]
        );
    }

    #[tokio::test]
    async fn last_prefill_chunks_are_batched_for_sampling() {
        let (mut runtime, state) = test_runtime_with_config(test_config_with_paragraph_len(2));
        let entry_id_1 = Uuid::new_v4();
        let entry_id_2 = Uuid::new_v4();
        insert_waiting_entry(&mut runtime, entry_id_1, vec![1, 2], 1, None);
        insert_waiting_entry(&mut runtime, entry_id_2, vec![3, 4], 1, None);

        assert!(runtime.tick_once().await);
        assert!(!runtime.entries.contains_key(&entry_id_1));
        assert!(!runtime.entries.contains_key(&entry_id_2));
        assert_eq!(
            *state.forward_calls.lock().expect("forward calls"),
            vec![(true, 2)]
        );
        assert_eq!(
            *state.reset_batch_indices.lock().expect("reset calls"),
            vec![0, 1]
        );
    }

    #[tokio::test]
    async fn cancelling_waiting_stream_entry_sends_error_and_removes_entry() {
        let (mut runtime, state) = test_runtime();
        let entry_id = Uuid::new_v4();
        let (stream_tx, mut stream_rx) = mpsc::channel(8);
        insert_waiting_entry(&mut runtime, entry_id, vec![11], 1, Some(stream_tx));

        runtime.cancel_entries(&[entry_id]).await;

        match timeout(Duration::from_secs(1), stream_rx.recv())
            .await
            .expect("cancel stream timeout")
        {
            Some(EngineEvent::Error(message)) => assert_eq!(message, "request cancelled"),
            other => panic!("unexpected cancel event: {other:?}"),
        }
        assert!(!runtime.entries.contains_key(&entry_id));
        assert!(!runtime.scheduler.has_waiting());
        assert!(!runtime.scheduler.has_running());
        assert!(
            state
                .reset_batch_indices
                .lock()
                .expect("reset calls")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn cancelling_running_entry_resets_batch_and_removes_entry() {
        let (mut runtime, state) = test_runtime();
        let entry_id = Uuid::new_v4();
        insert_waiting_entry(&mut runtime, entry_id, vec![11], 1, None);

        let step = runtime.scheduler.schedule(&mut runtime.entries);
        assert_eq!(step.prefill_ids, vec![entry_id]);

        runtime.cancel_entries(&[entry_id]).await;

        assert!(!runtime.entries.contains_key(&entry_id));
        assert!(!runtime.scheduler.has_waiting());
        assert!(!runtime.scheduler.has_running());
        assert_eq!(
            *state.reset_batch_indices.lock().expect("reset calls"),
            vec![0]
        );
    }

    #[tokio::test]
    async fn run_wakes_on_channel_recv_and_processes_submit() {
        let state = TestForwardState::default();
        let (tx, rx) = mpsc::channel(8);
        let runtime = EngineRuntime::new(
            test_config(),
            test_tokenizer(),
            rx,
            Box::new(TestForward::new(state.clone(), 1)),
        );
        let task = tokio::spawn(runtime.run());

        let entry_id = Uuid::new_v4();
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(InferenceSubmitCommand::SubmitText {
            entry_id,
            input_text: "hello".to_string(),
            sampling: SamplingConfig {
                max_new_tokens: 1,
                ..SamplingConfig::default()
            },
            stop_suffixes: Vec::new(),
            token_logprobs: None,
            stream: true,
            submitted_at: Instant::now(),
            validate_ms: None,
            reply: reply_tx,
        })
        .await
        .expect("submit command");

        let reply = timeout(Duration::from_secs(1), reply_rx)
            .await
            .expect("submit reply timeout")
            .expect("submit reply dropped");

        let mut stream_rx = match reply {
            InferenceSubmitResult::Stream {
                entry_id: reply_entry_id,
                rx,
            } => {
                assert_eq!(reply_entry_id, entry_id);
                rx
            }
            other => panic!("unexpected submit reply: {other:?}"),
        };

        loop {
            match timeout(Duration::from_secs(1), stream_rx.recv())
                .await
                .expect("stream event timeout")
            {
                Some(EngineEvent::Output(_)) => {}
                Some(EngineEvent::Done(meta)) => {
                    assert_eq!(meta.reason, FinishReason::Length);
                    assert_eq!(meta.generated_tokens, 1);
                    break;
                }
                Some(EngineEvent::Error(message)) => {
                    panic!("unexpected stream error: {message}");
                }
                None => panic!("stream closed before done"),
            }
        }

        drop(tx);
        timeout(Duration::from_secs(1), task)
            .await
            .expect("runtime task timeout")
            .expect("runtime task failed");

        assert_eq!(
            *state.forward_calls.lock().expect("forward calls"),
            vec![(true, 1)]
        );
    }
}

pub type EngineRuntimeConfig = InferenceExecutionConfig;
pub type EngineRuntime = InferenceExecutionLoop;
