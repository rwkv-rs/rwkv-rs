use std::collections::VecDeque;

use crate::types::{EntryId, InferEntry, InferEntryState};

#[derive(Debug)]
pub struct SchedulerStep {
    pub decode_ids: Vec<EntryId>,
    pub prefill_ids: Vec<EntryId>,
}

#[derive(Debug)]
pub struct DefaultScheduler {
    max_batch_size: usize,
    waiting: VecDeque<EntryId>,
    running: Vec<EntryId>,
    batch_slots: Vec<Option<EntryId>>,
    decode_first: bool,
}

impl DefaultScheduler {
    pub fn new(max_batch_size: usize, decode_first: bool) -> Self {
        Self {
            max_batch_size,
            waiting: VecDeque::new(),
            running: Vec::new(),
            batch_slots: vec![None; max_batch_size],
            decode_first,
        }
    }

    pub fn push_waiting(&mut self, entry_id: EntryId) {
        self.waiting.push_back(entry_id);
    }

    pub fn on_done(&mut self, entry_id: EntryId) {
        self.running.retain(|id| *id != entry_id);
        for slot in &mut self.batch_slots {
            if slot.as_ref().is_some_and(|id| *id == entry_id) {
                *slot = None;
            }
        }
    }

    pub fn batch_slots(&self) -> &[Option<EntryId>] {
        &self.batch_slots
    }

    pub fn schedule(
        &mut self,
        entries: &mut std::collections::HashMap<EntryId, InferEntry>,
    ) -> SchedulerStep {
        // Fill empty batch positions from waiting queue.
        for batch_index in 0..self.max_batch_size {
            if self.batch_slots[batch_index].is_some() {
                continue;
            }
            let Some(entry_id) = self.waiting.pop_front() else {
                break;
            };
            self.batch_slots[batch_index] = Some(entry_id);
            self.running.push(entry_id);
            if let Some(entry) = entries.get_mut(&entry_id) {
                entry.batch_index = Some(batch_index);
                if entry.state == InferEntryState::Waiting {
                    entry.state = InferEntryState::RunningPrefill;
                }
            }
        }

        // Decide work groups.
        let mut decode_ids = Vec::new();
        let mut prefill_ids = Vec::new();
        for entry_id in self.running.iter().copied() {
            let Some(entry) = entries.get(&entry_id) else {
                continue;
            };
            match entry.state {
                InferEntryState::RunningDecode => decode_ids.push(entry_id),
                InferEntryState::RunningPrefill => prefill_ids.push(entry_id),
                _ => {}
            }
        }

        // "One decode + one prefill" per tick semantics is enforced by the engine; here we just group.
        if self.decode_first {
            SchedulerStep {
                decode_ids,
                prefill_ids,
            }
        } else {
            // Still return both; engine can choose ordering.
            SchedulerStep {
                decode_ids,
                prefill_ids,
            }
        }
    }
}
