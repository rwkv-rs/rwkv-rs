use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;

use crate::processor::{Step, StepOutcome, file::Writer};

static BOS_TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"<BOS_TOKEN>").unwrap());

pub struct RemoveSpecialTokenFormatter {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
}

impl RemoveSpecialTokenFormatter {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self { writer }
    }
}

impl Step for RemoveSpecialTokenFormatter {
    fn name(&self) -> &'static str {
        "RemoveSpecialToken"
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
                let cleaned = BOS_TOKEN_REGEX.replace_all(data.as_ref(), "").into_owned();

                StepOutcome::Keep(Cow::Owned(cleaned))
            })
            .collect()
    }
}
