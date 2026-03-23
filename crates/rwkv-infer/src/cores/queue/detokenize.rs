use std::{sync::Arc, thread};

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::mpsc;

use super::{END_TOKEN_ID, Queue, QueueItem, QueueItemStatus};

pub(super) struct DetokenizeTask {
    pub(super) item_id: usize,
    pub(super) token_id: i32,
}

pub(super) struct DetokenizeResult {
    pub(super) item_id: usize,
    pub(super) token_bytes: Vec<u8>,
}

pub(super) fn spawn_detokenize_worker(
    tokenizer: Arc<Tokenizer>,
) -> (
    mpsc::UnboundedSender<DetokenizeTask>,
    mpsc::UnboundedReceiver<DetokenizeResult>,
) {
    let (detokenize_sender, mut detokenize_task_receiver) =
        mpsc::unbounded_channel::<DetokenizeTask>();
    let (detokenize_result_sender, detokenize_receiver) =
        mpsc::unbounded_channel::<DetokenizeResult>();

    thread::spawn(move || {
        while let Some(task) = detokenize_task_receiver.blocking_recv() {
            let token_bytes = if task.token_id == END_TOKEN_ID || task.token_id < 0 {
                Vec::new()
            } else {
                tokenizer.token_bytes(task.token_id as u16).to_vec()
            };

            if detokenize_result_sender
                .send(DetokenizeResult {
                    item_id: task.item_id,
                    token_bytes,
                })
                .is_err()
            {
                break;
            }
        }
    });

    (detokenize_sender, detokenize_receiver)
}

impl Queue {
    pub(super) fn queue_detokenize_token(
        &mut self,
        item_id: usize,
        token_id: i32,
    ) -> Result<(), String> {
        let detokenize_sender = self.detokenize_sender.clone();
        let item = self
            .items
            .get_mut(&item_id)
            .expect("scheduled item_id must exist in queue");

        item.pending_detokenize_tasks += 1;
        if detokenize_sender
            .send(DetokenizeTask { item_id, token_id })
            .is_err()
        {
            item.pending_detokenize_tasks -= 1;
            return Err("detokenize worker closed".to_string());
        }

        Ok(())
    }

    pub(super) fn drain_detokenize_results(&mut self) {
        let mut removed_item_ids = Vec::new();

        while let Ok(result) = self.detokenize_receiver.try_recv() {
            let Some(item) = self.items.get_mut(&result.item_id) else {
                continue;
            };

            item.pending_detokenize_tasks = item.pending_detokenize_tasks.saturating_sub(1);
            item.detokenize_buffer
                .extend_from_slice(&result.token_bytes);

            if let Some((matched_at, _)) = find_stop_suffix(&item.detokenize_buffer, &item.stop_suffixes) {
                let emit_len = valid_utf8_prefix_len(&item.detokenize_buffer[..matched_at]);
                if !emit_text(item, emit_len) {
                    removed_item_ids.push(result.item_id);
                    continue;
                }

                removed_item_ids.push(result.item_id);
                continue;
            }

            if item.status == QueueItemStatus::Finished && item.pending_detokenize_tasks == 0 {
                if !emit_text(item, item.detokenize_buffer.len()) {
                    removed_item_ids.push(result.item_id);
                    continue;
                }

                removed_item_ids.push(result.item_id);
                continue;
            }

            let hold_bytes = max_stop_suffix_bytes(&item.stop_suffixes).saturating_sub(1);
            let emit_limit = item.detokenize_buffer.len().saturating_sub(hold_bytes);
            let emit_len = valid_utf8_prefix_len(&item.detokenize_buffer[..emit_limit]);

            if !emit_text(item, emit_len) {
                removed_item_ids.push(result.item_id);
            }
        }

        if !removed_item_ids.is_empty() {
            removed_item_ids.sort_unstable();
            removed_item_ids.dedup();
            self.remove(&removed_item_ids);
        }
    }
}

fn emit_text(item: &mut QueueItem, emit_len: usize) -> bool {
    if emit_len == 0 {
        return true;
    }

    let bytes: Vec<u8> = item.detokenize_buffer.drain(..emit_len).collect();
    let delta_text = String::from_utf8_lossy(&bytes).into_owned();
    item.completions_text.push_str(&delta_text);
    delta_text.is_empty() || item.completions_tx.try_send(delta_text).is_ok()
}

fn valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(err) => err.valid_up_to(),
    }
}

fn max_stop_suffix_bytes(stop_suffixes: &[String]) -> usize {
    stop_suffixes
        .iter()
        .map(|suffix| suffix.len())
        .max()
        .unwrap_or(0)
}

fn find_stop_suffix(bytes: &[u8], stop_suffixes: &[String]) -> Option<(usize, usize)> {
    let mut matched = None;

    for stop_suffix in stop_suffixes {
        let stop_bytes = stop_suffix.as_bytes();
        if stop_bytes.is_empty() || stop_bytes.len() > bytes.len() {
            continue;
        }

        let Some(start) = bytes
            .windows(stop_bytes.len())
            .position(|window| window == stop_bytes)
        else {
            continue;
        };

        match matched {
            None => matched = Some((start, stop_bytes.len())),
            Some((best_start, best_len)) => {
                if start < best_start || (start == best_start && stop_bytes.len() > best_len) {
                    matched = Some((start, stop_bytes.len()));
                }
            }
        }
    }

    matched
}
