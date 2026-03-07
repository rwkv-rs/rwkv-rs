use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::inference_core::{EntryId, PrefillStepKind, RequestPhase, RequestState};

#[derive(Debug)]
pub struct SchedulerStep {
    pub prefill_without_output_ids: Vec<EntryId>,
    pub prefill_ids: Vec<EntryId>,
    pub decode_ids: Vec<EntryId>,
}

impl SchedulerStep {
    pub fn has_work(&self) -> bool {
        !self.prefill_without_output_ids.is_empty()
            || !self.prefill_ids.is_empty()
            || !self.decode_ids.is_empty()
    }
}

#[derive(Debug)]
pub struct Scheduler {
    max_batch_size: usize,
    paragraph_len: usize,
    waiting: VecDeque<EntryId>,
    running: Vec<EntryId>,
    batch_slots: Vec<Option<EntryId>>,
}

impl Scheduler {
    pub fn new(max_batch_size: usize, paragraph_len: usize) -> Self {
        Self {
            max_batch_size,
            paragraph_len,
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
        self.batch_slots.iter().any(Option::is_none)
    }

    pub fn has_waiting_entries(&self, entries: &HashMap<EntryId, RequestState>) -> bool {
        self.waiting.iter().any(|entry_id| {
            entries.get(entry_id).is_some_and(|entry| {
                entry.phase == RequestPhase::Waiting && entry.batch_index.is_none()
            })
        })
    }

    pub fn has_ready_decode(&self, entries: &HashMap<EntryId, RequestState>) -> bool {
        self.running.iter().any(|entry_id| {
            entries.get(entry_id).is_some_and(|entry| {
                entry.phase == RequestPhase::RunningDecode && entry.batch_index.is_some()
            })
        })
    }

    pub fn has_ready_prefill(&self, entries: &HashMap<EntryId, RequestState>) -> bool {
        self.running.iter().any(|entry_id| {
            entries.get(entry_id).is_some_and(|entry| {
                entry.phase == RequestPhase::RunningPrefill && entry.batch_index.is_some()
            })
        })
    }

    pub fn schedule(&mut self, entries: &mut HashMap<EntryId, RequestState>) -> SchedulerStep {
        rwkv_bench::trace_lite_scope!("rwkv.infer.scheduler.schedule");

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
                if entry.phase == RequestPhase::Waiting {
                    entry.phase = RequestPhase::RunningPrefill;
                }
                if entry.scheduled_at.is_none() {
                    entry.scheduled_at = Some(Instant::now());
                }
            }
        }

        let mut prefill_without_output_ids = Vec::new();
        let mut prefill_ids = Vec::new();
        let mut decode_ids = Vec::new();

        for entry_id in self.running.iter().copied() {
            let Some(entry) = entries.get_mut(&entry_id) else {
                continue;
            };
            match entry.phase {
                RequestPhase::RunningDecode => decode_ids.push(entry_id),
                RequestPhase::RunningPrefill => match entry.next_prefill_step(self.paragraph_len) {
                    Some(PrefillStepKind::WithoutOutput) => {
                        prefill_without_output_ids.push(entry_id);
                    }
                    Some(PrefillStepKind::WithOutput) => {
                        prefill_ids.push(entry_id);
                    }
                    None => {
                        entry.phase = RequestPhase::RunningDecode;
                        decode_ids.push(entry_id);
                    }
                },
                _ => {}
            }
        }

        SchedulerStep {
            prefill_without_output_ids,
            prefill_ids,
            decode_ids,
        }
    }

    fn pop_next_waiting(&mut self, entries: &HashMap<EntryId, RequestState>) -> Option<EntryId> {
        while let Some(entry_id) = self.waiting.pop_front() {
            let Some(entry) = entries.get(&entry_id) else {
                continue;
            };
            if entry.phase == RequestPhase::Waiting && entry.batch_index.is_none() {
                return Some(entry_id);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::Scheduler;
    use crate::inference_core::{RequestPhase, RequestState, SamplingConfig};
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn on_done_removes_waiting_entry() {
        let entry_id = Uuid::new_v4();
        let mut scheduler = Scheduler::new(1, 16);
        scheduler.push_waiting(entry_id);

        scheduler.on_done(entry_id);

        assert!(scheduler.waiting.is_empty());
        assert!(scheduler.running.is_empty());
        assert!(scheduler.batch_slots.iter().all(Option::is_none));
    }

    #[test]
    fn schedule_skips_cancelled_waiting_entries() {
        let entry_id = Uuid::new_v4();
        let mut scheduler = Scheduler::new(1, 16);
        let mut entries = HashMap::new();
        let mut entry = RequestState::new(entry_id, SamplingConfig::default(), Vec::new(), None);
        entry.phase = RequestPhase::Cancelled;
        entries.insert(entry_id, entry);
        scheduler.push_waiting(entry_id);

        let step = scheduler.schedule(&mut entries);

        assert!(!step.has_work());
        assert!(scheduler.batch_slots.iter().all(Option::is_none));
        assert!(scheduler.running.is_empty());
        assert!(scheduler.waiting.is_empty());
    }

    #[test]
    fn last_prefill_chunk_is_explicitly_marked_for_output() {
        let entry_id = Uuid::new_v4();
        let mut scheduler = Scheduler::new(1, 4);
        let mut entries = HashMap::new();
        let mut entry = RequestState::new(entry_id, SamplingConfig::default(), Vec::new(), None);
        entry.input_token_ids = vec![1, 2, 3, 4, 5];
        entries.insert(entry_id, entry);
        scheduler.push_waiting(entry_id);

        let step = scheduler.schedule(&mut entries);
        assert_eq!(step.prefill_without_output_ids, vec![entry_id]);
        assert!(step.prefill_ids.is_empty());

        let entry = entries.get_mut(&entry_id).expect("entry");
        entry.prefill_chunk_cursor = 1;

        let step = scheduler.schedule(&mut entries);
        assert!(step.prefill_without_output_ids.is_empty());
        assert_eq!(step.prefill_ids, vec![entry_id]);
    }
}
