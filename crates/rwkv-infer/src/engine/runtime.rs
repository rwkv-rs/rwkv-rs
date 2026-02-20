use std::collections::HashMap;
use std::time::Duration;

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::scheduler::DefaultScheduler;
use crate::types::{
    EngineEvent, EntryId, FinishMetadata, FinishReason, InferEntry, InferEntryState, SamplingConfig,
};

#[cfg(feature = "trace-lite")]
macro_rules! trace_lite_scope {
    ($name:literal) => {
        rwkv_trace::tracy_scope!($name);
    };
}

#[cfg(not(feature = "trace-lite"))]
macro_rules! trace_lite_scope {
    ($name:literal) => {};
}

#[cfg(feature = "trace-full")]
macro_rules! trace_full_scope {
    ($name:literal) => {
        rwkv_trace::tracy_scope!($name);
    };
}

#[cfg(not(feature = "trace-full"))]
macro_rules! trace_full_scope {
    ($name:literal) => {};
}

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

/// Model executor hook used by the engine.
pub trait InferExecutor: Send + 'static {
    fn prefill(&mut self, batch_positions: &[(usize, &[i32], &[u8])]) -> crate::Result<()>;

    fn decode(
        &mut self,
        batch_positions: &[(usize, i32)],
        sampling: SamplingConfig,
    ) -> crate::Result<Vec<(usize, i32)>>;

    fn reset_batch_position(&mut self, batch_index: usize) -> crate::Result<()>;
}

struct PendingSubmit {
    entry_id: EntryId,
    input_text: String,
    sampling: SamplingConfig,
    stop_suffixes: Vec<String>,
    stream: bool,
    reply: oneshot::Sender<crate::engine::SubmitOutput>,
}

pub struct EngineRuntime {
    cfg: EngineRuntimeConfig,
    tokenizer: Tokenizer,
    scheduler: DefaultScheduler,
    entries: HashMap<EntryId, InferEntry>,
    rx: mpsc::Receiver<crate::engine::EngineCommand>,
    executor: Box<dyn InferExecutor>,
}

impl EngineRuntime {
    pub fn spawn(
        cfg: EngineRuntimeConfig,
        executor: Box<dyn InferExecutor>,
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
        executor: Box<dyn InferExecutor>,
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
                trace_lite_scope!("rwkv_infer.engine.runtime.run");
                while let Ok(cmd) = self.rx.try_recv() {
                    match cmd {
                        crate::engine::EngineCommand::SubmitText {
                            input_text,
                            sampling,
                            stop_suffixes,
                            stream,
                            reply,
                        } => pending_submits.push(PendingSubmit {
                            entry_id: Uuid::new_v4(),
                            input_text,
                            sampling,
                            stop_suffixes,
                            stream,
                            reply,
                        }),
                        crate::engine::EngineCommand::Cancel { entry_id } => {
                            self.handle_cancel(entry_id);
                        }
                    }
                }
                self.handle_submit_batch(pending_submits);
            }

            self.tick().await;
            tokio::time::sleep(tick_sleep).await;
        }
    }

    fn handle_submit_batch(&mut self, submits: Vec<PendingSubmit>) {
        trace_full_scope!("rwkv_infer.engine.runtime.submit");
        if submits.is_empty() {
            return;
        }

        let texts: Vec<String> = submits
            .iter()
            .map(|submit| submit.input_text.clone())
            .collect();
        let tokenized: Vec<Vec<u16>> = self.tokenizer.encode_batch(texts, false);

        for (submit, token_ids_u16) in submits.into_iter().zip(tokenized.into_iter()) {
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
        trace_full_scope!("rwkv_infer.engine.runtime.cancel");
        if let Some(entry) = self.entries.get_mut(&entry_id) {
            entry.state = InferEntryState::Cancelled;
            if let Some(batch_index) = entry.batch_index.take() {
                let _ = self.executor.reset_batch_position(batch_index);
            }
        }
        self.scheduler.on_done(entry_id);
    }

    async fn tick(&mut self) {
        let step = {
            trace_lite_scope!("rwkv_infer.engine.runtime.tick");
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
        trace_full_scope!("rwkv_infer.engine.runtime.prefill");
        if prefill_ids.is_empty() {
            return;
        }

        let paragraph_len = self.cfg.paragraph_len;
        let mut batch_positions: Vec<(usize, Vec<i32>, Vec<u8>)> = Vec::new();

        for entry_id in prefill_ids.iter().copied() {
            let Some(entry) = self.entries.get_mut(&entry_id) else {
                continue;
            };
            let Some(batch_index) = entry.batch_index else {
                continue;
            };

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

            let token_ids = entry.prefill_padded_token_ids[start..end].to_vec();
            let mut context_mask = vec![1u8; token_ids.len()];
            if entry.prefill_chunk_cursor == 0 && entry.prefill_pad_len > 0 {
                for i in 0..entry.prefill_pad_len.min(context_mask.len()) {
                    context_mask[i] = 0;
                }
            }

            batch_positions.push((batch_index, token_ids, context_mask));
            entry.prefill_chunk_cursor += 1;

            if is_last_chunk {
                entry.state = InferEntryState::RunningDecode;
                entry.prefill_padded_token_ids.clear();
                entry.prefill_pad_len = 0;
                entry.prefill_chunk_cursor = 0;
            }
        }

        if batch_positions.is_empty() {
            return;
        }

        let args: Vec<(usize, &[i32], &[u8])> = batch_positions
            .iter()
            .map(|(batch_index, token_ids, context_mask)| {
                (*batch_index, token_ids.as_slice(), context_mask.as_slice())
            })
            .collect();

        if let Err(err) = self.executor.prefill(&args) {
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

        let mut groups: HashMap<(u32, i32, u32, u32, u32, u32), Vec<EntryId>> = HashMap::new();
        for entry_id in decode_ids.iter().copied() {
            let Some(entry) = self.entries.get(&entry_id) else {
                continue;
            };
            let key = (
                entry.sampling.temperature.to_bits(),
                entry.sampling.top_k,
                entry.sampling.top_p.to_bits(),
                entry.sampling.presence_penalty.to_bits(),
                entry.sampling.repetition_penalty.to_bits(),
                entry.sampling.penalty_decay.to_bits(),
            );
            groups.entry(key).or_default().push(entry_id);
        }

        for entry_ids in groups.values() {
            let mut batch_inputs: Vec<(usize, i32)> = Vec::new();
            let mut batch_to_entry: HashMap<usize, EntryId> = HashMap::new();
            let mut sampling = SamplingConfig::default();

            for entry_id in entry_ids.iter().copied() {
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

                sampling = entry.sampling;
                batch_inputs.push((batch_index, last_token_id));
                batch_to_entry.insert(batch_index, entry_id);
            }

            if batch_inputs.is_empty() {
                continue;
            }

            let decode_out = {
                trace_full_scope!("rwkv_infer.engine.runtime.decode");
                sampling.max_new_tokens = 1;
                match self.executor.decode(&batch_inputs, sampling) {
                    Ok(out) => out,
                    Err(err) => {
                        let chain = err.format_chain();
                        log::error!("decode failed: {chain}");
                        self.fail_entries(entry_ids, format!("decode failed: {chain}"))
                            .await;
                        continue;
                    }
                }
            };

            for (batch_index, token_id) in decode_out {
                let Some(entry_id) = batch_to_entry.get(&batch_index).copied() else {
                    continue;
                };
                let Some(entry) = self.entries.get_mut(&entry_id) else {
                    continue;
                };

                entry.generated_token_ids.push(token_id);
                let bytes = self.tokenizer.token_bytes(token_id as u16);
                entry.generated_bytes.extend_from_slice(bytes);

                let (emit_text, matched_stop) = entry.emit_stream_text();
                let matched_stop_index = matched_stop.map(|stop| stop.index);
                let hit_stop_suffix = matched_stop_index.is_some();
                let hit_max_new_tokens = entry.generated_token_ids.len() >= entry.max_new_tokens;
                let finish_meta =
                    finish_metadata_for_entry(entry, matched_stop_index, hit_max_new_tokens);

                let tx = entry.stream_tx.clone();
                if let Some(tx) = tx {
                    if !emit_text.is_empty() {
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
                            let _ = tx.send(EngineEvent::Text(tail)).await;
                        }
                        let _ = tx.send(EngineEvent::Done(meta.clone())).await;
                    }
                }

                if let Some(meta) = finish_meta {
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
                        let _ = self.executor.reset_batch_position(finished_batch_index);
                    }
                    self.scheduler.on_done(entry_id);
                }
            }
        }
    }

    async fn fail_entries(&mut self, entry_ids: &[EntryId], message: String) {
        for entry_id in entry_ids.iter().copied() {
            let mut stream_tx = None;
            if let Some(entry) = self.entries.get_mut(&entry_id) {
                entry.state = InferEntryState::Failed;
                stream_tx = entry.stream_tx.clone();
                if let Some(batch_index) = entry.batch_index.take() {
                    let _ = self.executor.reset_batch_position(batch_index);
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
) -> Option<FinishMetadata> {
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
        });
    }

    if hit_max_new_tokens {
        return Some(FinishMetadata {
            reason: FinishReason::Length,
            matched_stop_suffix: None,
            matched_stop_suffix_index: None,
            max_new_tokens: entry.max_new_tokens,
            generated_tokens: entry.generated_token_ids.len(),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::finish_metadata_for_entry;
    use crate::types::{FinishReason, InferEntry, SamplingConfig};
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
        let meta = finish_metadata_for_entry(&entry, Some(0), true).expect("must finish");
        assert_eq!(meta.reason, FinishReason::Stop);
        assert_eq!(meta.matched_stop_suffix.as_deref(), Some("stop"));
        assert_eq!(meta.matched_stop_suffix_index, Some(0));
        assert_eq!(meta.generated_tokens, 1);
        assert_eq!(meta.max_new_tokens, 1);
    }

    #[test]
    fn finish_metadata_reports_length_when_no_stop_match() {
        let sampling = SamplingConfig {
            max_new_tokens: 2,
            ..SamplingConfig::default()
        };
        let mut entry = InferEntry::new(Uuid::new_v4(), "prompt".to_string(), sampling, Vec::new());
        entry.generated_token_ids.extend([1, 2]);
        let meta = finish_metadata_for_entry(&entry, None, true).expect("must finish");
        assert_eq!(meta.reason, FinishReason::Length);
        assert_eq!(meta.matched_stop_suffix, None);
        assert_eq!(meta.matched_stop_suffix_index, None);
        assert_eq!(meta.generated_tokens, 2);
        assert_eq!(meta.max_new_tokens, 2);
    }
}
