use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackgroundTaskState {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct BackgroundTask {
    pub task_id: String,
    pub response_id: String,
    pub state: BackgroundTaskState,
    pub error: Option<String>,
}

#[derive(Clone, Default)]
pub struct BackgroundTaskManager {
    inner: Arc<RwLock<HashMap<String, BackgroundTask>>>,
}

impl BackgroundTaskManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&self, response_id: String) -> BackgroundTask {
        let task = BackgroundTask {
            task_id: format!("task_{}", Uuid::new_v4().as_simple()),
            response_id,
            state: BackgroundTaskState::Queued,
            error: None,
        };

        if let Ok(mut guard) = self.inner.write() {
            guard.insert(task.task_id.clone(), task.clone());
        }
        task
    }

    pub fn get(&self, task_id: &str) -> Option<BackgroundTask> {
        self.inner
            .read()
            .ok()
            .and_then(|guard| guard.get(task_id).cloned())
    }

    pub fn set_state(&self, task_id: &str, state: BackgroundTaskState) {
        if let Ok(mut guard) = self.inner.write() {
            if let Some(task) = guard.get_mut(task_id) {
                task.state = state;
            }
        }
    }

    pub fn fail(&self, task_id: &str, message: String) {
        if let Ok(mut guard) = self.inner.write() {
            if let Some(task) = guard.get_mut(task_id) {
                task.state = BackgroundTaskState::Failed;
                task.error = Some(message);
            }
        }
    }
}

pub static GLOBAL_BACKGROUND_TASKS: Lazy<BackgroundTaskManager> =
    Lazy::new(BackgroundTaskManager::new);
