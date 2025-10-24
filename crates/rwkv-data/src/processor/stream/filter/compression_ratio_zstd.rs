use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use rayon::prelude::*;

use crate::processor::{Step, StepOutcome, file::Writer};

pub struct ZstdCompressionFilterStep {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
}

impl ZstdCompressionFilterStep {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self { writer }
    }

    fn should_filter(&self, text: &str) -> bool {
        const MIN_RATIO: f64 = 0.18; // Low compression ratio -> excessive repetition
        const MAX_RATIO: f64 = 0.92; // High compression ratio -> near random/hash
        const LEVEL: i32 = 3; // Light compression level
        const MIN_TEXT_BYTES: usize = 256; // Skip compression check for too short text
        if text.is_empty() {
            return true;
        }

        let payload = text.as_bytes();

        let original_size = payload.len();

        if original_size < MIN_TEXT_BYTES {
            return false;
        }

        let compressed = zstd::encode_all(payload, LEVEL).unwrap();

        let ratio = compressed.len() as f64 / original_size as f64;

        ratio < MIN_RATIO || ratio > MAX_RATIO
    }
}

impl Step for ZstdCompressionFilterStep {
    fn name(&self) -> &'static str {
        "ZstdCompressionFilter"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        batch
            .into_par_iter()
            .map(|data| {
                if self.should_filter(data.as_ref()) {
                    StepOutcome::Exclude(data)
                } else {
                    StepOutcome::Keep(data)
                }
            })
            .collect()
    }
}
