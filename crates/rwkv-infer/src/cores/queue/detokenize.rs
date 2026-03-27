use std::{sync::Arc, thread};

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::mpsc;

use super::{
    END_TOKEN_ID,
    Queue,
    QueueEvent,
    QueueFinishMeta,
    QueueFinishReason,
    QueueItem,
    QueueItemStatus,
    QueueOutput,
    QueueOutputCandidate,
    QueueOutputToken,
};
use crate::cores::forward::{Logprob, TokenId};

pub(super) struct DetokenizeTask {
    pub(super) item_id: usize,
    pub(super) token: TokenId,
}

pub(super) struct DetokenizeResult {
    pub(super) item_id: usize,
    pub(super) token: QueueOutputToken,
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
            if detokenize_result_sender
                .send(DetokenizeResult {
                    item_id: task.item_id,
                    token: build_output_token(&tokenizer, task.token),
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
        token: TokenId,
    ) -> Result<(), String> {
        let detokenize_sender = self.detokenize_sender.clone();
        let item = self
            .items
            .get_mut(&item_id)
            .expect("scheduled item_id must exist in queue");

        item.pending_detokenize_tasks += 1;
        if detokenize_sender
            .send(DetokenizeTask { item_id, token })
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
            let mut batch_id_to_reset = None;
            let mut should_remove = false;
            let mut stop_match = None;
            let mut should_finish = false;

            {
                let Some(item) = self.items.get_mut(&result.item_id) else {
                    continue;
                };

                item.pending_detokenize_tasks = item.pending_detokenize_tasks.saturating_sub(1);

                let stable_output = push_utf8_output(item, result.token);
                let (delta, matched_stop_suffix_index) = push_stop_output(item, stable_output);
                if !emit_output(item, delta) {
                    should_remove = true;
                } else if let Some(index) = matched_stop_suffix_index {
                    let finish_meta = item.finish_meta.get_or_insert(QueueFinishMeta {
                        reason: QueueFinishReason::Stop,
                        matched_stop_suffix: None,
                        matched_stop_suffix_index: None,
                        generated_tokens: item.generated_tokens(),
                    });
                    finish_meta.reason = QueueFinishReason::Stop;
                    finish_meta.matched_stop_suffix = Some(item.stop_suffixes[index].0.clone());
                    finish_meta.matched_stop_suffix_index = Some(index);
                    item.status = QueueItemStatus::Finished;
                    batch_id_to_reset = item.batch_id.take();
                    stop_match = Some(index);
                } else if item.status == QueueItemStatus::Finished
                    && item.pending_detokenize_tasks == 0
                {
                    should_finish = true;
                }
            }

            if let Some(batch_id) = batch_id_to_reset {
                self.model_forward.reset(batch_id);
            }

            if should_remove {
                removed_item_ids.push(result.item_id);
                continue;
            }

            if stop_match.is_some() {
                let item = self
                    .items
                    .get_mut(&result.item_id)
                    .expect("finished item must exist");
                let finish_meta = item.finish_meta.take().expect("missing finish meta");
                let _ = item.completions_tx.try_send(QueueEvent::Done(finish_meta));
                removed_item_ids.push(result.item_id);
                continue;
            }

            if should_finish && self.finish_item_if_ready(result.item_id) {
                removed_item_ids.push(result.item_id);
            }
        }

        if !removed_item_ids.is_empty() {
            removed_item_ids.sort_unstable();
            removed_item_ids.dedup();
            self.remove(&removed_item_ids);
        }
    }

    pub(super) fn finish_item_if_ready(&mut self, item_id: usize) -> bool {
        let Some(item) = self.items.get_mut(&item_id) else {
            return false;
        };
        if item.status != QueueItemStatus::Finished || item.pending_detokenize_tasks != 0 {
            return false;
        }

        let final_delta = finish_output(item);
        if !emit_output(item, final_delta) {
            return true;
        }

        let finish_meta = item.finish_meta.take().expect("missing finish meta");
        let _ = item.completions_tx.try_send(QueueEvent::Done(finish_meta));
        true
    }
}

fn emit_output(item: &mut QueueItem, delta: QueueOutput) -> bool {
    if delta.text.is_empty() && delta.tokens.is_empty() {
        return true;
    }

    item.completions_tx
        .try_send(QueueEvent::Delta(delta))
        .is_ok()
}

fn push_utf8_output(item: &mut QueueItem, token: QueueOutputToken) -> QueueOutput {
    item.tokens_in_buffer.push(token);

    let emit_count = longest_valid_utf8_token_prefix(&item.tokens_in_buffer);
    if emit_count == 0 {
        return QueueOutput::default();
    }

    let tokens: Vec<_> = item.tokens_in_buffer.drain(..emit_count).collect();
    QueueOutput {
        text: decode_tokens_text(&tokens),
        tokens,
    }
}

fn push_stop_output(item: &mut QueueItem, output: QueueOutput) -> (QueueOutput, Option<usize>) {
    if output.tokens.is_empty() {
        return (QueueOutput::default(), None);
    }

    item.pending_stop_tokens.extend(output.tokens);
    let pending_bytes = collect_token_bytes(&item.pending_stop_tokens);

    if let Some((matched_at, matched_stop_suffix_index)) =
        find_stop_suffix(&pending_bytes, &item.stop_suffixes)
    {
        return (
            take_stop_output(item, matched_at, true),
            Some(matched_stop_suffix_index),
        );
    }

    let emit_limit = pending_bytes
        .len()
        .saturating_sub(item.max_stop_suffix_len.saturating_sub(1));
    (take_stop_output(item, emit_limit, false), None)
}

fn finish_output(item: &mut QueueItem) -> QueueOutput {
    let emit_limit = collect_token_bytes(&item.pending_stop_tokens).len();
    take_stop_output(item, emit_limit, false)
}

fn take_stop_output(
    item: &mut QueueItem,
    emit_limit: usize,
    allow_partial_text: bool,
) -> QueueOutput {
    if emit_limit == 0 || item.pending_stop_tokens.is_empty() {
        return QueueOutput::default();
    }

    let mut emit_count = 0;
    let mut emitted_bytes = 0;
    for token in &item.pending_stop_tokens {
        let next = emitted_bytes + token.bytes.len();
        if next > emit_limit {
            break;
        }
        emitted_bytes = next;
        emit_count += 1;
    }

    let tokens: Vec<_> = item.pending_stop_tokens.drain(..emit_count).collect();
    let mut text = decode_tokens_text(&tokens);

    if allow_partial_text && emitted_bytes < emit_limit {
        if let Some(token) = item.pending_stop_tokens.first() {
            let partial_len = emit_limit - emitted_bytes;
            let partial_bytes = &token.bytes[..partial_len.min(token.bytes.len())];
            let valid_prefix_len = longest_valid_utf8_prefix_len(partial_bytes);
            if valid_prefix_len > 0 {
                text.push_str(&String::from_utf8_lossy(&partial_bytes[..valid_prefix_len]));
            }
        }
    }

    QueueOutput { text, tokens }
}

fn build_output_token(tokenizer: &Tokenizer, token: TokenId) -> QueueOutputToken {
    let bytes = token_bytes(tokenizer, token.token_id);

    QueueOutputToken {
        token: String::from_utf8_lossy(&bytes).into_owned(),
        bytes,
        logprob: token.logprob.as_ref().map(|logprob| logprob.logprob),
        top_logprobs: build_top_logprobs(tokenizer, token.logprob.as_ref()),
    }
}

fn build_top_logprobs(
    tokenizer: &Tokenizer,
    logprob: Option<&Logprob>,
) -> Vec<QueueOutputCandidate> {
    let Some(logprob) = logprob else {
        return Vec::new();
    };

    logprob
        .top_logprobs
        .iter()
        .map(|candidate| {
            let bytes = token_bytes(tokenizer, candidate.token_id);
            QueueOutputCandidate {
                token: String::from_utf8_lossy(&bytes).into_owned(),
                bytes,
                logprob: candidate.logprob,
            }
        })
        .collect()
}

fn token_bytes(tokenizer: &Tokenizer, token_id: i32) -> Vec<u8> {
    if token_id == END_TOKEN_ID || token_id < 0 {
        Vec::new()
    } else {
        tokenizer.token_bytes(token_id as u16).to_vec()
    }
}

fn collect_token_bytes(tokens: &[QueueOutputToken]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for token in tokens {
        bytes.extend_from_slice(&token.bytes);
    }
    bytes
}

fn decode_tokens_text(tokens: &[QueueOutputToken]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    String::from_utf8_lossy(&collect_token_bytes(tokens)).into_owned()
}

fn longest_valid_utf8_token_prefix(tokens: &[QueueOutputToken]) -> usize {
    if tokens.is_empty() {
        return 0;
    }

    let mut buf = Vec::new();
    let mut last_valid = 0;
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

fn longest_valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(err) => err.valid_up_to(),
    }
}

fn find_stop_suffix(bytes: &[u8], stop_suffixes: &[(String, Vec<u8>)]) -> Option<(usize, usize)> {
    let mut matched = None;

    for (suffix_index, (_, stop_suffix)) in stop_suffixes.iter().enumerate() {
        if stop_suffix.is_empty() || stop_suffix.len() > bytes.len() {
            continue;
        }

        let Some(start) = bytes
            .windows(stop_suffix.len())
            .position(|window| window == stop_suffix)
        else {
            continue;
        };

        match matched {
            None => matched = Some((start, suffix_index, stop_suffix.len())),
            Some((best_start, _, best_len)) => {
                if start < best_start || (start == best_start && stop_suffix.len() > best_len) {
                    matched = Some((start, suffix_index, stop_suffix.len()));
                }
            }
        }
    }

    matched.map(|(start, suffix_index, _)| (start, suffix_index))
}
