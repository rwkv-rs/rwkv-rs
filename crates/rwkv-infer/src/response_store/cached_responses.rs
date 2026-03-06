use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct CachedResponse {
    pub response_id: String,
    pub output_text: String,
    pub output_token_ids: Vec<i32>,
}

#[derive(Clone, Default)]
pub struct ResponseCache {
    inner: Arc<RwLock<HashMap<String, CachedResponse>>>,
}

impl ResponseCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&self, response: CachedResponse) {
        if let Ok(mut guard) = self.inner.write() {
            guard.insert(response.response_id.clone(), response);
        }
    }

    pub fn get(&self, response_id: &str) -> Option<CachedResponse> {
        self.inner
            .read()
            .ok()
            .and_then(|guard| guard.get(response_id).cloned())
    }

    pub fn remove(&self, response_id: &str) -> bool {
        self.inner
            .write()
            .ok()
            .and_then(|mut guard| guard.remove(response_id))
            .is_some()
    }

    pub fn next_response_id(&self) -> String {
        format!("resp_{}", Uuid::new_v4().as_simple())
    }
}

pub static GLOBAL_RESPONSE_CACHE: Lazy<ResponseCache> = Lazy::new(ResponseCache::new);
