use std::collections::HashSet;

use super::{BatchStatus, Queue, QueueItem, QueueItemStatus, guided_decode::GuidedDecodingStatus};

impl Queue {
    pub(super) fn assign_batch_ids(&mut self, item_ids: &[usize]) {
        let resident_count = self.resident_count();
        let mut used_batch_ids: HashSet<usize> = self
            .items
            .values()
            .filter_map(|item| item.batch_id)
            .collect();
        let new_item_ids: Vec<_> = item_ids
            .iter()
            .copied()
            .filter(|item_id| {
                self.items
                    .get(item_id)
                    .expect("scheduled item_id must exist in queue")
                    .batch_id
                    .is_none()
            })
            .collect();
        let free_slot_count = self.max_batch_size.saturating_sub(resident_count);

        assert!(
            new_item_ids.len() <= free_slot_count,
            "scheduler selected too many new items for free batch slots: max_batch_size={}, resident_count={}, free_slot_count={}, newly_scheduled_count={}",
            self.max_batch_size,
            resident_count,
            free_slot_count,
            new_item_ids.len(),
        );

        for item_id in new_item_ids {
            let batch_id = (0..self.max_batch_size)
                .find(|candidate| !used_batch_ids.contains(candidate))
                .expect("free batch slot must exist for newly scheduled item");

            self.set_guided_token_mask_row(batch_id, None);
            self.items
                .get_mut(&item_id)
                .expect("scheduled item_id must exist in queue")
                .batch_id = Some(batch_id);
            used_batch_ids.insert(batch_id);
        }
    }

    pub(super) fn select_next_batch(&self) -> Option<(BatchStatus, Vec<usize>)> {
        if self.decode_slot_count() < self.max_batch_size {
            if let Some(batch) = self.select_prefill_batch() {
                return Some(batch);
            }
        }

        let decode_item_ids = self.collect_step_item_ids(BatchStatus::Decode);
        if !decode_item_ids.is_empty() {
            return Some((BatchStatus::Decode, decode_item_ids));
        }

        self.select_prefill_batch()
    }

    pub(super) fn infer_step_batch_status(&self, item_ids: &[usize]) -> Option<BatchStatus> {
        if item_ids.is_empty() || item_ids.len() > self.max_batch_size {
            return None;
        }

        let mut seen_item_ids = HashSet::with_capacity(item_ids.len());
        let mut batch_status = None;
        let mut new_item_count = 0;

        for item_id in item_ids {
            if !seen_item_ids.insert(*item_id) {
                return None;
            }

            let item = self.items.get(item_id)?;
            let (item_batch_status, _) = ready_step_group(item)?;

            match batch_status {
                Some(expected_status) if expected_status != item_batch_status => return None,
                Some(_) => {}
                None => batch_status = Some(item_batch_status),
            }

            if item.batch_id.is_none() {
                new_item_count += 1;
            }
        }

        let batch_status = batch_status?;
        if batch_status != BatchStatus::Decode && new_item_count > self.free_slot_count() {
            return None;
        }

        Some(batch_status)
    }

    fn select_prefill_batch(&self) -> Option<(BatchStatus, Vec<usize>)> {
        for batch_status in [BatchStatus::Prefill, BatchStatus::PrefillWithoutOutput] {
            let item_ids = self.collect_step_item_ids(batch_status);
            if !item_ids.is_empty() {
                return Some((batch_status, item_ids));
            }
        }

        None
    }

    pub(super) fn collect_step_item_ids(&self, batch_status: BatchStatus) -> Vec<usize> {
        let mut item_ids = Vec::with_capacity(self.max_batch_size);

        item_ids.extend(
            self.items
                .iter()
                .filter(|(_, item)| {
                    item.batch_id.is_some() && ready_step_group(item) == Some((batch_status, 0))
                })
                .map(|(item_id, _)| *item_id),
        );

        if batch_status != BatchStatus::Decode {
            let new_item_limit = self
                .free_slot_count()
                .min(self.max_batch_size.saturating_sub(item_ids.len()));
            item_ids.extend(
                self.items
                    .iter()
                    .filter(|(_, item)| {
                        item.batch_id.is_none() && ready_step_group(item) == Some((batch_status, 1))
                    })
                    .take(new_item_limit)
                    .map(|(item_id, _)| *item_id),
            );
        }

        item_ids
    }

    fn resident_count(&self) -> usize {
        self.items
            .values()
            .filter(|item| item.batch_id.is_some())
            .count()
    }

    fn free_slot_count(&self) -> usize {
        self.max_batch_size.saturating_sub(self.resident_count())
    }

    fn decode_slot_count(&self) -> usize {
        self.items
            .values()
            .filter(|item| {
                item.batch_id.is_some() && matches!(item.status, QueueItemStatus::Decode(_))
            })
            .count()
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

    match step_group.0 {
        BatchStatus::PrefillWithoutOutput => Some(step_group),
        BatchStatus::Prefill | BatchStatus::Decode => match item.guided_decoding_status {
            GuidedDecodingStatus::Pending => None,
            GuidedDecodingStatus::Disabled | GuidedDecodingStatus::Ready(_) => Some(step_group),
        },
    }
}
