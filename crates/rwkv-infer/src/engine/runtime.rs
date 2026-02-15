use std::collections::HashMap;
use std::time::Duration;

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::config::{BackendConfig, SamplingConfig};
use crate::scheduler::DefaultScheduler;
use crate::types::{EngineEvent, InferEntry};
use crate::types::{EntryId, InferEntryState};

#[derive(Clone, Debug)]
pub struct EngineRuntimeConfig {
    pub backend: BackendConfig,
}

impl Default for EngineRuntimeConfig {
    fn default() -> Self {
        Self {
            backend: BackendConfig::default(),
        }
    }
}

/// Model executor hook used by the engine.
///
/// This keeps rwkv-infer independent from a specific model crate. The first
/// integration is expected to live in `examples/rwkv-lm`.
pub trait InferExecutor: Send + 'static {
    fn tokenize(&self, text: &str) -> crate::Result<Vec<i32>>;
    fn detokenize(&self, token_ids: &[i32]) -> crate::Result<String>;

    fn prefill(&mut self, batch_positions: &[(usize, &[i32], &[u8])]) -> crate::Result<()>;

    fn decode(
        &mut self,
        batch_positions: &[(usize, i32)],
        sampling: SamplingConfig,
    ) -> crate::Result<Vec<(usize, i32)>>;

    fn reset_batch_position(&mut self, batch_index: usize) -> crate::Result<()>;
}

pub struct EngineRuntime {
    cfg: EngineRuntimeConfig,
    scheduler: DefaultScheduler,
    entries: HashMap<EntryId, InferEntry>,
    rx: mpsc::Receiver<crate::engine::EngineCommand>,
    executor: Box<dyn InferExecutor>,
}

impl EngineRuntime {
    pub fn spawn(
        cfg: EngineRuntimeConfig,
        executor: Box<dyn InferExecutor>,
    ) -> crate::engine::EngineHandle {
        let (tx, rx) = mpsc::channel(1024);
        let handle = crate::engine::EngineHandle::new(tx);
        let rt = Self::new(cfg, rx, executor);
        tokio::spawn(async move {
            rt.run().await;
        });
        handle
    }

    pub fn new(
        cfg: EngineRuntimeConfig,
        rx: mpsc::Receiver<crate::engine::EngineCommand>,
        executor: Box<dyn InferExecutor>,
    ) -> Self {
        let scheduler = DefaultScheduler::new(cfg.backend.max_batch_size, cfg.backend.decode_first);
        Self {
            cfg,
            scheduler,
            entries: HashMap::new(),
            rx,
            executor,
        }
    }

    pub async fn run(mut self) {
        let tick_sleep = Duration::from_millis(1);
        loop {
            // Drain commands quickly.
            while let Ok(cmd) = self.rx.try_recv() {
                self.handle_command(cmd).await;
            }

            self.tick().await;
            tokio::time::sleep(tick_sleep).await;
        }
    }

    async fn handle_command(&mut self, cmd: crate::engine::EngineCommand) {
        use crate::engine::{EngineCommand, SubmitOutput};
        match cmd {
            EngineCommand::SubmitText {
                input_text,
                sampling,
                stream,
                reply,
            } => {
                let entry_id = Uuid::new_v4();
                let mut entry = InferEntry::new(entry_id, input_text, sampling);

                match self.executor.tokenize(&entry.input_text) {
                    Ok(token_ids) => {
                        if token_ids.len() > self.cfg.backend.max_context_length {
                            let _ = reply.send(SubmitOutput::Error {
                                entry_id,
                                message: format!(
                                    "prompt too long: {} tokens > max_context_length={}",
                                    token_ids.len(),
                                    self.cfg.backend.max_context_length
                                ),
                            });
                            return;
                        }
                        entry.input_token_ids = token_ids;
                    }
                    Err(e) => {
                        let _ = reply.send(SubmitOutput::Error {
                            entry_id,
                            message: e.to_string(),
                        });
                        return;
                    }
                }

                if stream {
                    let (tx, rx) = mpsc::channel(256);
                    entry.stream_tx = Some(tx);
                    self.entries.insert(entry_id, entry);
                    self.scheduler.push_waiting(entry_id);
                    let _ = reply.send(SubmitOutput::Stream { entry_id, rx });
                } else {
                    // Non-streaming: store the entry and let the caller poll via responses API later.
                    self.entries.insert(entry_id, entry);
                    self.scheduler.push_waiting(entry_id);
                    let _ = reply.send(SubmitOutput::Done {
                        entry_id,
                        output_text: String::new(),
                        token_ids: Vec::new(),
                    });
                }
            }
            EngineCommand::Cancel { entry_id } => {
                if let Some(entry) = self.entries.get_mut(&entry_id) {
                    entry.state = InferEntryState::Cancelled;
                    if let Some(batch_index) = entry.batch_index.take() {
                        let _ = self.executor.reset_batch_position(batch_index);
                    }
                }
                self.scheduler.on_done(entry_id);
            }
        }
    }

    async fn tick(&mut self) {
        let step = self.scheduler.schedule(&mut self.entries);

        if self.cfg.backend.decode_first {
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

        let chunk = self.cfg.backend.prefill_chunk_size;
        let mut batch_positions: Vec<(usize, Vec<i32>, Vec<u8>)> = Vec::new();

        for entry_id in prefill_ids.iter().copied() {
            let Some(entry) = self.entries.get_mut(&entry_id) else {
                continue;
            };
            let Some(batch_index) = entry.batch_index else {
                continue;
            };

            if entry.prefill_padded_token_ids.is_empty() {
                // Left-pad to a multiple of `chunk`.
                let len = entry.input_token_ids.len();
                let pad_len = (chunk - (len % chunk)) % chunk;
                entry.prefill_pad_len = pad_len;
                entry.prefill_padded_token_ids = std::iter::repeat(0)
                    .take(pad_len)
                    .chain(entry.input_token_ids.iter().copied())
                    .collect();
                entry.prefill_chunk_cursor = 0;
            }

            let padded_len = entry.prefill_padded_token_ids.len();
            let start = entry.prefill_chunk_cursor * chunk;
            if start >= padded_len {
                entry.state = InferEntryState::RunningDecode;
                continue;
            }
            let end = (start + chunk).min(padded_len);
            let is_last_chunk = end == padded_len;

            let tok = entry.prefill_padded_token_ids[start..end].to_vec();
            let mut mask = vec![1u8; tok.len()];
            if entry.prefill_chunk_cursor == 0 && entry.prefill_pad_len > 0 {
                for i in 0..entry.prefill_pad_len.min(mask.len()) {
                    mask[i] = 0;
                }
            }

            batch_positions.push((batch_index, tok, mask));
            entry.prefill_chunk_cursor += 1;

            if is_last_chunk {
                // Next tick can directly decode; avoid an extra "empty prefill" tick.
                entry.state = InferEntryState::RunningDecode;
                entry.prefill_padded_token_ids.clear();
                entry.prefill_pad_len = 0;
                entry.prefill_chunk_cursor = 0;
            }
        }

        if batch_positions.is_empty() {
            return;
        }

        // Execute in a single call; the executor is responsible for batching efficiently.
        let args: Vec<(usize, &[i32], &[u8])> = batch_positions
            .iter()
            .map(|(i, t, m)| (*i, t.as_slice(), m.as_slice()))
            .collect();
        if let Err(e) = self.executor.prefill(&args) {
            log::error!("prefill failed: {e}");
        }
    }

    async fn run_decode(&mut self, decode_ids: &[EntryId]) {
        if decode_ids.is_empty() {
            return;
        }

        // Group by sampling hyperparams so we can keep decode batched.
        let mut groups: HashMap<(u32, i32, u32), Vec<EntryId>> = HashMap::new();
        for entry_id in decode_ids.iter().copied() {
            let Some(entry) = self.entries.get(&entry_id) else {
                continue;
            };
            let key = (
                entry.sampling.temperature.to_bits(),
                entry.sampling.top_k,
                entry.sampling.top_p.to_bits(),
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

                // Use the last token as decode input; if no generated tokens yet, use last prompt token.
                let last = entry
                    .generated_token_ids
                    .last()
                    .copied()
                    .or_else(|| entry.input_token_ids.last().copied())
                    .unwrap_or(0);

                sampling = entry.sampling;
                batch_inputs.push((batch_index, last));
                batch_to_entry.insert(batch_index, entry_id);
            }

            if batch_inputs.is_empty() {
                continue;
            }

            // Decode is always 1 token per tick; stopping is controlled by `entry.max_new_tokens`.
            sampling.max_new_tokens = 1;

            let out = match self.executor.decode(&batch_inputs, sampling) {
                Ok(out) => out,
                Err(e) => {
                    log::error!("decode failed: {e}");
                    continue;
                }
            };

            for (batch_index, token_id) in out {
                let Some(entry_id) = batch_to_entry.get(&batch_index).copied() else {
                    continue;
                };
                let Some(entry) = self.entries.get_mut(&entry_id) else {
                    continue;
                };

                entry.generated_token_ids.push(token_id);
                let text = self.executor.detokenize(&[token_id]).unwrap_or_default();
                if let Some(tx) = entry.stream_tx.as_ref() {
                    let _ = tx.send(EngineEvent::Text(text)).await;
                }

                if entry.generated_token_ids.len() >= entry.max_new_tokens {
                    entry.state = InferEntryState::Done;
                    if let Some(tx) = entry.stream_tx.as_ref() {
                        let _ = tx.send(EngineEvent::Done).await;
                    }
                    if let Some(batch_index) = entry.batch_index.take() {
                        let _ = self.executor.reset_batch_position(batch_index);
                    }
                    self.scheduler.on_done(entry_id);
                }
            }
        }
    }
}
