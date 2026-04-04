use std::collections::HashMap;

use crate::cores::forward::{StepMode, TokenId, TokenIdLogprobsConfig};
use super::{
    BatchStatus,
    END_TOKEN_ID,
    Queue,
    QueueFinishMeta,
    QueueFinishReason,
    QueueItemStatus,
    guided_decode::GuidedDecodingStatus,
};

pub(super) struct StepInputs {
    pub(super) batch_ids: Vec<usize>,
    pub(super) contexts: Vec<Vec<i32>>,
    pub(super) context_masks: Vec<Vec<u8>>,
    pub(super) sampling_configs: Vec<crate::cores::forward::sampling::SamplingConfig>,
    pub(super) token_logprobs_configs: Vec<Option<TokenIdLogprobsConfig>>,
    pub(super) has_masked_guided_token: bool,
}

impl Queue {
    pub(super) fn build_step_inputs(&self, item_ids: &[usize]) -> StepInputs {
        let mut batch_ids = Vec::with_capacity(item_ids.len());
        let mut contexts = Vec::with_capacity(item_ids.len());
        let mut context_masks = Vec::with_capacity(item_ids.len());
        let mut sampling_configs = Vec::with_capacity(item_ids.len());
        let mut token_logprobs_configs = Vec::with_capacity(item_ids.len());
        let mut has_masked_guided_token = false;

        for item_id in item_ids {
            let item = self
                .items
                .get(item_id)
                .expect("scheduled item_id must exist in queue");
            let batch_id = item
                .batch_id
                .expect("batch_id should be assigned before forward");
            batch_ids.push(batch_id);

            let (context, context_mask) = match self.batch_status {
                BatchStatus::PrefillWithoutOutput | BatchStatus::Prefill => {
                    let next_paragraph_id = match item.status {
                        QueueItemStatus::Waiting => 0,
                        QueueItemStatus::Prefill(next_paragraph_id) => next_paragraph_id,
                        _ => panic!("prefill item status"),
                    };

                    let start = next_paragraph_id * self.paragraph_len;
                    let end = start + self.paragraph_len;
                    let context = item.context_tokens_for_step[start..end].to_vec();

                    let leading_zero_count = context
                        .iter()
                        .take_while(|&&token_id| token_id == 0)
                        .count();
                    let mut context_mask = vec![1u8; context.len()];
                    for index in 0..leading_zero_count.saturating_sub(1) {
                        context_mask[index] = 0;
                    }

                    (context, context_mask)
                }
                BatchStatus::Decode => (
                    item.context_tokens_for_step.clone(),
                    vec![1u8; item.context_tokens_for_step.len()],
                ),
            };

            contexts.push(context);
            context_masks.push(context_mask);
            sampling_configs.push(item.sampling_config);
            token_logprobs_configs.push(item.token_logprobs_config.clone());
            match &item.guided_decoding_status {
                GuidedDecodingStatus::Disabled => {}
                GuidedDecodingStatus::Pending => {
                    panic!("guided decoding step scheduled before prepared state was ready")
                }
                GuidedDecodingStatus::Ready(prepared_state) => {
                    has_masked_guided_token |= prepared_state.has_masked_token();
                }
            }
        }

        StepInputs {
            batch_ids,
            contexts,
            context_masks,
            sampling_configs,
            token_logprobs_configs,
            has_masked_guided_token,
        }
    }

    pub(super) fn advance_prefill_items(&mut self, item_ids: &[usize]) {
        for item_id in item_ids {
            let item = self
                .items
                .get_mut(item_id)
                .expect("scheduled item_id must exist in queue");

            item.status = match item.status {
                QueueItemStatus::Waiting => QueueItemStatus::Prefill(1),
                QueueItemStatus::Prefill(next_paragraph_id) => {
                    QueueItemStatus::Prefill(next_paragraph_id + 1)
                }
                _ => panic!("prefill item status"),
            };
        }
    }

    pub(super) fn apply_output_tokens(&mut self, item_ids: &[usize], new_tokens: Vec<TokenId>) {
        let from_prefill = self.batch_status == BatchStatus::Prefill;
        let token_by_batch_id: HashMap<usize, TokenId> = new_tokens
            .into_iter()
            .map(|token| (token.batch_index, token))
            .collect();

        let mut removed_item_ids = Vec::new();

        for &item_id in item_ids {
            let (batch_id, next_len, max_new_tokens) = {
                let item = self
                    .items
                    .get(&item_id)
                    .expect("scheduled item_id must exist in queue");

                let next_len = if from_prefill {
                    1
                } else {
                    match item.status {
                        QueueItemStatus::Decode(old_len) => old_len + 1,
                        _ => panic!("decode item status"),
                    }
                };

                (
                    item.batch_id.expect("scheduled item must have batch_id"),
                    next_len,
                    item.sampling_config.max_new_tokens,
                )
            };
            let Some(token) = token_by_batch_id.get(&batch_id) else {
                removed_item_ids.push(item_id);
                continue;
            };

            let token_id = token.token_id;
            let finished_by_model = token_id == END_TOKEN_ID || token.finish_after_token;
            let finished_by_length = next_len >= max_new_tokens;
            let finished_by_guided = match self.apply_guided_prepared_state(
                item_id,
                token_id,
                !(finished_by_model || finished_by_length),
            ) {
                Ok(finished_by_guided) => finished_by_guided,
                Err(_) => {
                    removed_item_ids.push(item_id);
                    continue;
                }
            };
            let should_finish = finished_by_model || finished_by_guided || finished_by_length;

            if token_id != END_TOKEN_ID
                && self.queue_detokenize_token(item_id, token.clone()).is_err()
            {
                removed_item_ids.push(item_id);
                continue;
            }

            let finish_reason = if should_finish {
                Some(if finished_by_model || finished_by_guided {
                    QueueFinishReason::Stop
                } else if finished_by_length {
                    QueueFinishReason::Length
                } else {
                    QueueFinishReason::Stop
                })
            } else {
                None
            };

            let (batch_id_to_reset, should_remove) = {
                let item = self
                    .items
                    .get_mut(&item_id)
                    .expect("scheduled item_id must exist in queue");

                if should_finish {
                    item.finish_meta = Some(QueueFinishMeta {
                        reason: finish_reason.expect("finish reason"),
                        matched_stop_suffix: None,
                        matched_stop_suffix_index: None,
                        generated_tokens: next_len,
                    });
                    item.status = QueueItemStatus::Finished;
                    (item.batch_id.take(), item.pending_detokenize_tasks == 0)
                } else {
                    item.status = QueueItemStatus::Decode(next_len);
                    item.context_tokens_for_step = vec![token_id];
                    (None, false)
                }
            };

            if let Some(batch_id) = batch_id_to_reset {
                self.model_forward.reset(batch_id);
            }
            if should_remove && self.finish_item_if_ready(item_id) {
                removed_item_ids.push(item_id);
            }
        }

        if !removed_item_ids.is_empty() {
            removed_item_ids.sort_unstable();
            removed_item_ids.dedup();
            self.remove(&removed_item_ids);
        }
    }

    pub(super) fn build_step_mode<'a>(
        batch_status: BatchStatus,
        step_inputs: &'a StepInputs,
        guided_token_mask_ref: Option<crate::cores::forward::GuidedTokenMaskBatchRef<'a>>,
    ) -> StepMode<'a> {
        match batch_status {
            BatchStatus::PrefillWithoutOutput => StepMode::PrefillNoOutput,
            BatchStatus::Prefill | BatchStatus::Decode => StepMode::Sample {
                sampling_configs: &step_inputs.sampling_configs,
                token_logprobs_configs: &step_inputs.token_logprobs_configs,
                guided_token_mask_ref,
            },
        }
    }
}
