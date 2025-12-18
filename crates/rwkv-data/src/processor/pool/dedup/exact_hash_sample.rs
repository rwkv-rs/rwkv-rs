use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use dashmap::DashSet;
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_128;

use crate::processor::{Step, StepOutcome, file::Writer};

pub struct ExactHashSampleDedup {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    seen_hashes: DashSet<u128>,
}

impl ExactHashSampleDedup {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            seen_hashes: DashSet::new(),
        }
    }
}

impl Step for ExactHashSampleDedup {
    fn name(&self) -> &'static str {
        "ExactHashSampleDedup"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        let hashed: Vec<(Cow<'static, str>, u128)> = batch
            .into_par_iter()
            .map(|data| {
                let hash = xxh3_128(data.as_bytes());

                (data, hash)
            })
            .collect();

        hashed
            .into_iter()
            .map(|(data, hash)| {
                if self.seen_hashes.insert(hash) {
                    StepOutcome::Keep(data)
                } else {
                    StepOutcome::Exclude(data)
                }
            })
            .collect()
    }
}
