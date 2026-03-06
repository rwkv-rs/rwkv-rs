use std::collections::VecDeque;
use std::time::Instant;

use crate::inference_core::EntryId;
use crate::inference_core::{ActiveRequest as InferEntry, ActiveRequestState as InferEntryState};

#[derive(Debug)]
pub struct SchedulerStep {
    pub decode_ids: Vec<EntryId>,
    pub prefill_ids: Vec<EntryId>,
}

impl SchedulerStep {
    pub fn has_work(&self) -> bool {
        !self.decode_ids.is_empty() || !self.prefill_ids.is_empty()
    }
}

#[derive(Debug)]
pub struct DefaultScheduler {
    max_batch_size: usize,
    waiting: VecDeque<EntryId>,
    running: Vec<EntryId>,
    batch_slots: Vec<Option<EntryId>>,
}

impl DefaultScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            waiting: VecDeque::new(),
            running: Vec::new(),
            batch_slots: vec![None; max_batch_size],
        }
    }

    pub fn push_waiting(&mut self, entry_id: EntryId) {
        self.waiting.push_back(entry_id);
    }

    pub fn on_done(&mut self, entry_id: EntryId) {
        self.waiting.retain(|id| *id != entry_id);
        self.running.retain(|id| *id != entry_id);
        for slot in &mut self.batch_slots {
            if slot.as_ref().is_some_and(|id| *id == entry_id) {
                *slot = None;
            }
        }
    }

    pub fn has_waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    pub fn has_running(&self) -> bool {
        !self.running.is_empty()
    }

    pub fn has_free_slot(&self) -> bool {
        self.batch_slots.iter().any(|slot| slot.is_none())
    }

    pub fn has_waiting_entries(
        &self,
        entries: &std::collections::HashMap<EntryId, InferEntry>,
    ) -> bool {
        self.waiting.iter().any(|entry_id| {
            entries.get(entry_id).is_some_and(|entry| {
                entry.state == InferEntryState::Waiting && entry.batch_index.is_none()
            })
        })
    }

    pub fn has_ready_decode(
        &self,
        entries: &std::collections::HashMap<EntryId, InferEntry>,
    ) -> bool {
        self.running.iter().any(|entry_id| {
            entries.get(entry_id).is_some_and(|entry| {
                entry.state == InferEntryState::RunningDecode && entry.batch_index.is_some()
            })
        })
    }

    pub fn has_ready_prefill(
        &self,
        entries: &std::collections::HashMap<EntryId, InferEntry>,
    ) -> bool {
        self.running.iter().any(|entry_id| {
            entries.get(entry_id).is_some_and(|entry| {
                entry.state == InferEntryState::RunningPrefill && entry.batch_index.is_some()
            })
        })
    }

    pub fn batch_slots(&self) -> &[Option<EntryId>] {
        &self.batch_slots
    }

    pub fn schedule(
        &mut self,
        entries: &mut std::collections::HashMap<EntryId, InferEntry>,
    ) -> SchedulerStep {
        rwkv_bench::trace_lite_scope!("rwkv.infer.scheduler.default.schedule");
        #[cfg(feature = "trace")]
        tracing::trace!(
            waiting = self.waiting.len(),
            running = self.running.len(),
            capacity = self.max_batch_size,
            "scheduler tick"
        );

        for batch_index in 0..self.max_batch_size {
            if self.batch_slots[batch_index].is_some() {
                continue;
            }
            let Some(entry_id) = self.pop_next_waiting(entries) else {
                break;
            };
            self.batch_slots[batch_index] = Some(entry_id);
            self.running.push(entry_id);
            if let Some(entry) = entries.get_mut(&entry_id) {
                entry.batch_index = Some(batch_index);
                if entry.state == InferEntryState::Waiting {
                    entry.state = InferEntryState::RunningPrefill;
                }
                if entry.scheduled_at.is_none() {
                    entry.scheduled_at = Some(Instant::now());
                }
            }
        }

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

        #[cfg(feature = "trace")]
        tracing::trace!(
            decode = decode_ids.len(),
            prefill = prefill_ids.len(),
            occupied_slots = self
                .batch_slots
                .iter()
                .filter(|slot| slot.is_some())
                .count(),
            waiting = self.waiting.len(),
            "scheduler groups built"
        );

        SchedulerStep {
            decode_ids,
            prefill_ids,
        }
    }

    fn pop_next_waiting(
        &mut self,
        entries: &std::collections::HashMap<EntryId, InferEntry>,
    ) -> Option<EntryId> {
        while let Some(entry_id) = self.waiting.pop_front() {
            let Some(entry) = entries.get(&entry_id) else {
                continue;
            };
            if entry.state == InferEntryState::Waiting && entry.batch_index.is_none() {
                return Some(entry_id);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::DefaultScheduler;
    use crate::inference_core::ActiveRequest as InferEntry;
    use crate::inference_core::SamplingConfig;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn on_done_removes_waiting_entry() {
        let entry_id = Uuid::new_v4();
        let mut scheduler = DefaultScheduler::new(1);
        scheduler.push_waiting(entry_id);

        scheduler.on_done(entry_id);

        assert!(scheduler.waiting.is_empty());
        assert!(scheduler.running.is_empty());
        assert!(scheduler.batch_slots.iter().all(|slot| slot.is_none()));
    }

    #[test]
    fn schedule_skips_cancelled_waiting_entries() {
        let entry_id = Uuid::new_v4();
        let mut scheduler = DefaultScheduler::new(1);
        let mut entries = HashMap::new();
        let mut entry = InferEntry::new(
            entry_id,
            "prompt".to_string(),
            SamplingConfig::default(),
            Vec::new(),
            None,
        );
        entry.state = crate::inference_core::ActiveRequestState::Cancelled;
        entries.insert(entry_id, entry);
        scheduler.push_waiting(entry_id);

        let step = scheduler.schedule(&mut entries);

        assert!(!step.has_work());
        assert!(scheduler.batch_slots.iter().all(|slot| slot.is_none()));
        assert!(scheduler.running.is_empty());
        assert!(scheduler.waiting.is_empty());
    }
}
