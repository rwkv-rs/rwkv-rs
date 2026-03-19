use std::collections::BTreeMap;
use std::sync::Arc;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use tokio::sync::Semaphore;

use crate::db::CompletionKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AttemptKey {
    pub sample_index: usize,
    pub avg_repeat_index: usize,
    pub pass_index: u8,
}

impl From<CompletionKey> for AttemptKey {
    fn from(value: CompletionKey) -> Self {
        Self {
            sample_index: value.sample_index as usize,
            avg_repeat_index: value.avg_repeat_index as usize,
            pass_index: value.pass_index as u8,
        }
    }
}

pub(crate) struct AttemptOutcome {
    pub key: AttemptKey,
    pub is_passed: bool,
}

pub(crate) struct PendingChecker {
    pub completions_id: i32,
    pub context: String,
    pub answer: String,
    pub ref_answer: String,
}

pub(crate) struct CheckerRuntime {
    pub model_name: String,
    pub client: Arc<Client<OpenAIConfig>>,
    pub semaphore: Arc<Semaphore>,
}

pub(crate) struct TaskExecutionState {
    pub task_id: Option<i32>,
    pub results: BTreeMap<AttemptKey, bool>,
    pub pending_checks: Vec<PendingChecker>,
}
