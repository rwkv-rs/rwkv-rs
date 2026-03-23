use std::collections::HashSet;

use super::{BatchStatus, Queue, QueueItem, QueueItemStatus};

impl Queue {
    pub(super) fn assign_batch_ids(&mut self, item_ids: &[usize]) {
        let mut used_batch_ids: HashSet<usize> = self
            .items
            .values()
            .filter_map(|item| item.batch_id)
            .collect();

        for item_id in item_ids {
            let item = self
                .items
                .get_mut(item_id)
                .expect("scheduled item_id must exist in queue");
            if item.batch_id.is_some() {
                continue;
            }

            let batch_id = (0..self.max_batch_size)
                .find(|candidate| !used_batch_ids.contains(candidate))
                .expect("scheduler should not select more items than free batch slots");

            item.batch_id = Some(batch_id);
            used_batch_ids.insert(batch_id);
        }
    }

    pub(super) fn update_batch_status(&mut self) {
        if self.items.is_empty() {
            self.batch_status = BatchStatus::PrefillWithoutOutput;
            return;
        }

        let mut has_prefill_without_output = false;
        let mut has_prefill = false;
        let mut has_decode = false;

        for item in self.items.values() {
            match ready_step_group(item).map(|(batch_status, _)| batch_status) {
                Some(BatchStatus::PrefillWithoutOutput) => has_prefill_without_output = true,
                Some(BatchStatus::Prefill) => has_prefill = true,
                Some(BatchStatus::Decode) => has_decode = true,
                None => {}
            }
        }

        self.batch_status = match self.batch_status {
            BatchStatus::PrefillWithoutOutput if has_decode => BatchStatus::Decode,
            BatchStatus::PrefillWithoutOutput if has_prefill => BatchStatus::Prefill,
            BatchStatus::Prefill if has_decode => BatchStatus::Decode,
            BatchStatus::Prefill if has_prefill_without_output => BatchStatus::PrefillWithoutOutput,
            BatchStatus::Prefill if has_prefill => BatchStatus::Prefill,
            BatchStatus::Decode if has_prefill => BatchStatus::Prefill,
            BatchStatus::Decode if has_prefill_without_output => BatchStatus::PrefillWithoutOutput,
            BatchStatus::Decode if has_decode => BatchStatus::Decode,
            _ => BatchStatus::PrefillWithoutOutput,
        };
    }

    pub(super) fn collect_step_item_ids(&self) -> Vec<usize> {
        let mut item_ids = Vec::with_capacity(self.max_batch_size);

        for priority in 0..=usize::from(self.batch_status != BatchStatus::Decode) {
            item_ids.extend(
                self.items
                    .iter()
                    .filter(|(_, item)| ready_step_group(item) == Some((self.batch_status, priority)))
                    .take(self.max_batch_size - item_ids.len())
                    .map(|(item_id, _)| *item_id),
            );

            if item_ids.len() == self.max_batch_size {
                break;
            }
        }

        item_ids
    }
}

fn step_group(item: &QueueItem) -> Option<(BatchStatus, usize)> {
    match item.status {
        QueueItemStatus::Waiting if item.num_paragraphs > 1 => {
            Some((BatchStatus::PrefillWithoutOutput, 1))
        }
        QueueItemStatus::Waiting => Some((BatchStatus::Prefill, 1)),
        QueueItemStatus::Prefill(next_paragraph_id)
            if next_paragraph_id + 1 < item.num_paragraphs =>
        {
            Some((BatchStatus::PrefillWithoutOutput, 0))
        }
        QueueItemStatus::Prefill(_) => Some((BatchStatus::Prefill, 0)),
        QueueItemStatus::Decode(new_tokens_len)
            if new_tokens_len < item.sampling_config.max_new_tokens =>
        {
            Some((BatchStatus::Decode, 0))
        }
        QueueItemStatus::Decode(_) | QueueItemStatus::Finished => None,
    }
}

fn ready_step_group(item: &QueueItem) -> Option<(BatchStatus, usize)> {
    let step_group = step_group(item)?;

    if step_group.0 == BatchStatus::PrefillWithoutOutput
        || !uses_guided_decoding(item)
        || item.guided_token_mask.is_some()
    {
        Some(step_group)
    } else {
        None
    }
}

fn uses_guided_decoding(item: &QueueItem) -> bool {
    item.guided_decoding_config.is_some()
        || item.guided_decoding_pending
        || item.guided_decoding_state.is_some()
        || item.guided_token_mask.is_some()
}
