use std::collections::HashMap;
use std::time::{Duration, Instant};

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};

use crate::scheduler::DefaultScheduler;
use crate::types::{
    EngineEvent, EntryId, FinishMetadata, FinishReason, InferEntry, InferEntryState, SamplingConfig,
};

#[derive(Clone, Debug)]
pub struct EngineRuntimeConfig {
    pub tokenizer_vocab_path: String,
    pub max_batch_size: usize,
    pub paragraph_len: usize,
    pub max_context_len: usize,
    pub decode_first: bool,
}

impl Default for EngineRuntimeConfig {
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
    /// - `batch_positions`: `[(batch_index, &[token_ids], &[context_mask_u8])]`
    ///   prefill chunk: `token_ids.len() == paragraph_len`
    ///   decode step: `token_ids.len() == 1`
    /// - `samplings`: per-position sampling parameters (one per batch_position)
    /// - `need_sample`: when false, skip unembed+sampling (prefill intermediate chunk), return empty Vec
    /// Returns: `[(batch_index, sampled_token_id)]`
    fn forward(
        &mut self,
        batch_positions: &[(usize, &[i32], &[u8])],
        samplings: &[SamplingConfig],
        need_sample: bool,
    ) -> crate::Result<Vec<(usize, i32)>>;

    /// Reset recurrent state for the given batch position.
    fn reset(&mut self, batch_index: usize) -> crate::Result<()>;
}

struct PendingSubmit {
    entry_id: EntryId,
    input_text: String,
    sampling: SamplingConfig,
    stop_suffixes: Vec<String>,
    stream: bool,
    submitted_at: Instant,
    runtime_received_at: Instant,
    validate_ms: Option<u64>,
    reply: oneshot::Sender<crate::engine::SubmitOutput>,
}

pub struct EngineRuntime {
    cfg: EngineRuntimeConfig,
    tokenizer: Tokenizer,
    scheduler: DefaultScheduler,
    entries: HashMap<EntryId, InferEntry>,
    rx: mpsc::Receiver<crate::engine::EngineCommand>,
    executor: Box<dyn ModelForward>,
}

impl EngineRuntime {
    pub fn spawn(
        cfg: EngineRuntimeConfig,
        executor: Box<dyn ModelForward>,
    ) -> crate::Result<crate::engine::EngineHandle> {
        let tokenizer = Tokenizer::new(&cfg.tokenizer_vocab_path).map_err(|e| {
            crate::Error::bad_request(format!(
                "failed to load tokenizer vocab {}: {e}",
                cfg.tokenizer_vocab_path
            ))
        })?;

        let (tx, rx) = mpsc::channel(1024);
        let handle = crate::engine::EngineHandle::new(tx);
        let rt = Self::new(cfg, tokenizer, rx, executor);
        tokio::spawn(async move {
            rt.run().await;
        });
        Ok(handle)
    }

    pub fn new(
        cfg: EngineRuntimeConfig,
        tokenizer: Tokenizer,
        rx: mpsc::Receiver<crate::engine::EngineCommand>,
        executor: Box<dyn ModelForward>,
    ) -> Self {
        let scheduler = DefaultScheduler::new(cfg.max_batch_size, cfg.decode_first);
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
        let tick_sleep = Duration::from_millis(1);
        loop {
            let mut pending_submits = Vec::new();
            {
                rwkv_bench::trace_lite_scope!("rwkv.infer.engine.runtime.run");
                while let Ok(cmd) = self.rx.try_recv() {
                    match cmd {
                        crate::engine::EngineCommand::SubmitText {
                            entry_id,
                            input_text,
                            sampling,
                            stop_suffixes,
                            stream,
                            submitted_at,
                            validate_ms,
                            reply,
                        } => pending_submits.push(PendingSubmit {
                            entry_id,
                            input_text,
                            sampling,
                            stop_suffixes,
                            stream,
                            submitted_at,
                            runtime_received_at: Instant::now(),
                            validate_ms,
                            reply,
                        }),
                        crate::engine::EngineCommand::Cancel { entry_id } => {
                            self.handle_cancel(entry_id);
                        }
                    }
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

            self.tick().await;
            tokio::time::sleep(tick_sleep).await;
        }
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
                let _ = submit.reply.send(crate::engine::SubmitOutput::Error {
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
                let _ = submit.reply.send(crate::engine::SubmitOutput::Stream {
                    entry_id: submit.entry_id,
                    rx,
                });
            } else {
                self.entries.insert(submit.entry_id, entry);
                self.scheduler.push_waiting(submit.entry_id);
                let _ = submit.reply.send(crate::engine::SubmitOutput::Done {
                    entry_id: submit.entry_id,
                    output_text: String::new(),
                    token_ids: Vec::new(),
                });
            }
        }
    }

    fn handle_cancel(&mut self, entry_id: EntryId) {
        rwkv_bench::trace_scope!("rwkv.infer.engine.runtime.cancel");
        if let Some(entry) = self.entries.get_mut(&entry_id) {
            entry.state = InferEntryState::Cancelled;
            if let Some(batch_index) = entry.batch_index.take() {
                let _ = self.executor.reset(batch_index);
            }
        }
        self.scheduler.on_done(entry_id);
    }

    async fn tick(&mut self) {
        let step = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.engine.runtime.tick");
            self.scheduler.schedule(&mut self.entries)
        };

        if self.cfg.decode_first {
            self.run_decode(&step.decode_ids).await;
            self.run_prefill(&step.prefill_ids).await;
        } else {
            self.run_prefill(&step.prefill_ids).await;
            self.run_decode(&step.decode_ids).await;
        }
    }

    async fn run_prefill(&mut self, prefill_ids: &[EntryId]) {
        if prefill_ids.is_empty() {
            return;
        }
        let prefill_result = {
            rwkv_bench::trace_scope!("rwkv.infer.engine.runtime.prefill");
            let paragraph_len = self.cfg.paragraph_len;
            let mut batch_positions: Vec<(usize, Vec<i32>, Vec<u8>)> = Vec::new();
            let mut last_chunk_entries: Vec<EntryId> = Vec::new();
            let mut last_chunk_samplings: Vec<SamplingConfig> = Vec::new();

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
                    entry.prefill_padded_token_ids = std::iter::repeat(0)
                        .take(pad_len)
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
                    paragraph_len
                )
                .entered();

                let token_ids = entry.prefill_padded_token_ids[start..end].to_vec();
                let mut context_mask = vec![1u8; token_ids.len()];
                if entry.prefill_chunk_cursor == 0 && entry.prefill_pad_len > 0 {
                    for i in 0..entry.prefill_pad_len.min(context_mask.len()) {
                        context_mask[i] = 0;
                    }
                }

                batch_positions.push((batch_index, token_ids, context_mask));
                entry.prefill_chunk_cursor += 1;
                entry.last_prefill_at = Some(chunk_issued_at);

                if is_last_chunk {
                    entry.state = InferEntryState::RunningDecode;
                    entry.prefill_padded_token_ids.clear();
                    entry.prefill_pad_len = 0;
                    entry.prefill_chunk_cursor = 0;
                    last_chunk_entries.push(entry_id);
                    last_chunk_samplings.push(entry.sampling);
                }
            }

            if batch_positions.is_empty() {
                return;
            }

            #[cfg(feature = "trace")]
            tracing::info!(
                stage = "prefill_step",
                batch_size = batch_positions.len(),
                "dispatch prefill batch"
            );

            let args: Vec<(usize, &[i32], &[u8])> = batch_positions
                .iter()
                .map(|(batch_index, token_ids, context_mask)| {
                    (*batch_index, token_ids.as_slice(), context_mask.as_slice())
                })
                .collect();
            #[cfg(feature = "nsys")]
            let _nvtx_prefill = nvtx::range!("rwkv.infer.prefill");
            // For now, prefill all chunks without sampling.
            // Last-chunk sampling will be handled by the next decode tick.
            let default_samplings = vec![SamplingConfig::default(); args.len()];
            self.executor.forward(&args, &default_samplings, false)
        };

        if let Err(err) = prefill_result {
            let chain = err.format_chain();
            log::error!("prefill failed: {chain}");
            self.fail_entries(prefill_ids, format!("prefill failed: {chain}"))
                .await;
        }
    }

    async fn run_decode(&mut self, decode_ids: &[EntryId]) {
        if decode_ids.is_empty() {
            return;
        }

        let mut batch_inputs: Vec<(usize, i32)> = Vec::new();
        let mut batch_to_entry: HashMap<usize, EntryId> = HashMap::new();
        let mut samplings: Vec<SamplingConfig> = Vec::new();

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

            samplings.push(entry.sampling);
            batch_inputs.push((batch_index, last_token_id));
            batch_to_entry.insert(batch_index, entry_id);
        }

        if batch_inputs.is_empty() {
            return;
        }

        #[cfg(feature = "trace")]
        tracing::info!(
            target: "rwkv.infer",
            stage = "decode_step",
            batch_size = batch_inputs.len(),
            "dispatch decode batch"
        );

        let decode_result = {
            rwkv_bench::trace_scope!("rwkv.infer.engine.runtime.decode");
            #[cfg(feature = "nsys")]
            let _nvtx_decode = nvtx::range!("rwkv.infer.decode");
            let args: Vec<(usize, Vec<i32>, Vec<u8>)> = batch_inputs
                .iter()
                .map(|(batch_index, token_id)| (*batch_index, vec![*token_id], vec![1u8]))
                .collect();
            let args_ref: Vec<(usize, &[i32], &[u8])> = args
                .iter()
                .map(|(bi, tids, cm)| (*bi, tids.as_slice(), cm.as_slice()))
                .collect();
            self.executor.forward(&args_ref, &samplings, true)
        };
        let decode_out = match decode_result {
            Ok(out) => out,
            Err(err) => {
                let chain = err.format_chain();
                log::error!("decode failed: {chain}");
                self.fail_entries(decode_ids, format!("decode failed: {chain}"))
                    .await;
                return;
            }
        };

        for (batch_index, token_id) in decode_out {
            let Some(entry_id) = batch_to_entry.get(&batch_index).copied() else {
                continue;
            };
            let Some(entry) = self.entries.get_mut(&entry_id) else {
                continue;
            };
            let decode_at = Instant::now();
            if entry.first_decode_at.is_none() {
                entry.first_decode_at = Some(decode_at);
            }
            entry.last_decode_at = Some(decode_at);

            #[cfg(feature = "trace")]
            tracing::trace!(
                target: "rwkv.infer",
                request_id = %entry.entry_id,
                batch_index,
                generated_tokens = entry.generated_token_ids.len(),
                "decode token"
            );

            entry.generated_token_ids.push(token_id);
            let bytes = self.tokenizer.token_bytes(token_id as u16);
            entry.generated_bytes.extend_from_slice(bytes);

            let (emit_text, matched_stop) = entry.emit_stream_text();
            if !emit_text.is_empty() && entry.first_emit_at.is_none() {
                entry.first_emit_at = Some(Instant::now());
            }
            let matched_stop_index = matched_stop.map(|stop| stop.index);
            let hit_stop_suffix = matched_stop_index.is_some();
            let hit_max_new_tokens = entry.generated_token_ids.len() >= entry.max_new_tokens;
            let finish_meta = finish_metadata_for_entry(
                entry,
                matched_stop_index,
                hit_max_new_tokens,
                Instant::now(),
            );

            let tx = entry.stream_tx.clone();
            if let Some(tx) = tx {
                if !emit_text.is_empty() {
                    #[cfg(feature = "trace")]
                    tracing::trace!(
                        request_id = %entry.entry_id,
                        chars = emit_text.chars().count(),
                        "stream emit chunk"
                    );
                    let _ = tx.send(EngineEvent::Text(emit_text)).await;
                }

                if let Some(meta) = finish_meta.as_ref() {
                    let flush_lossy = false;
                    let tail = if hit_stop_suffix {
                        entry.flush_stream_text_until(entry.stop_trunc_len(), flush_lossy)
                    } else {
                        entry.flush_stream_text_until(entry.generated_bytes.len(), flush_lossy)
                    };
                    if !tail.is_empty() {
                        if entry.first_emit_at.is_none() {
                            entry.first_emit_at = Some(Instant::now());
                        }
                        let _ = tx.send(EngineEvent::Text(tail)).await;
                    }
                    let _ = tx.send(EngineEvent::Done(meta.clone())).await;
                }
            }

            if let Some(meta) = finish_meta {
                #[cfg(feature = "trace")]
                tracing::info!(
                    request_id = %entry.entry_id,
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
                entry.state = InferEntryState::Done;
                if let Some(finished_batch_index) = entry.batch_index.take() {
                    let _ = self.executor.reset(finished_batch_index);
                }
                self.scheduler.on_done(entry_id);
            }
        }
    }

    async fn fail_entries(&mut self, entry_ids: &[EntryId], message: String) {
        for entry_id in entry_ids.iter().copied() {
            #[cfg(feature = "trace")]
            tracing::error!(request_id = %entry_id, error = %message, "entry failed");
            let mut stream_tx = None;
            if let Some(entry) = self.entries.get_mut(&entry_id) {
                entry.state = InferEntryState::Failed;
                stream_tx = entry.stream_tx.clone();
                if let Some(batch_index) = entry.batch_index.take() {
                    let _ = self.executor.reset(batch_index);
                }
            }
            if let Some(tx) = stream_tx {
                let _ = tx.send(EngineEvent::Error(message.clone())).await;
            }
            self.scheduler.on_done(entry_id);
        }
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

fn build_timing_breakdown(
    entry: &InferEntry,
    finished_at: Instant,
) -> Option<crate::types::TimingBreakdownMs> {
    let timings = crate::types::TimingBreakdownMs {
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
    use super::finish_metadata_for_entry;
    use crate::types::{FinishReason, InferEntry, SamplingConfig};
    use std::time::Instant;
    use uuid::Uuid;

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
        let mut entry = InferEntry::new(Uuid::new_v4(), "prompt".to_string(), sampling, Vec::new());
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
}
