use std::{
    borrow::Cow,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use rayon::prelude::*;

use crate::{
    processor::{Step, StepOutcome, file::Writer},
    tokenizer::Tokenizer,
};

pub struct TokenizerCompressionFilterStep {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    vocab_path: String,
    tokenizer: Mutex<Option<Arc<Tokenizer>>>,
}

impl TokenizerCompressionFilterStep {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            vocab_path: String::new(),
            tokenizer: Mutex::new(None),
        }
    }

    pub fn set_vocab_path(&mut self, vocab_path: String) {
        self.vocab_path = vocab_path;

        if let Ok(mut guard) = self.tokenizer.lock() {
            *guard = None;
        }
    }

    fn tokenizer(&self) -> Arc<Tokenizer> {
        let vocab_path = self.vocab_path.clone();

        let mut guard = self.tokenizer.lock().unwrap();

        guard
            .get_or_insert_with(|| {
                assert!(
                    !vocab_path.is_empty(),
                    "Tokenizer vocab path must be configured before running the pipeline"
                );

                Arc::new(Tokenizer::new(&vocab_path).expect("failed to load tokenizer vocab"))
            })
            .clone()
    }

    fn should_filter(&self, text: &str, tokenizer: &Tokenizer) -> bool {
        const MIN_RATIO: f64 = 0.08; // Low compression ratio -> excessive repetition or very long tokens
        const MAX_RATIO: f64 = 0.85; // High compression ratio -> too many special chars or poor tokenization
        const MIN_TEXT_LENGTH: usize = 50; // Skip compression check for too short text
        if text.is_empty() {
            return true;
        }

        let char_count = text.chars().count();

        if char_count < MIN_TEXT_LENGTH {
            return false;
        }

        let token_ids = tokenizer.encode(text, false);

        let token_count = token_ids.len();

        if token_count == 0 {
            return true;
        }

        let ratio = token_count as f64 / char_count as f64;

        ratio < MIN_RATIO || ratio > MAX_RATIO
    }
}

impl Step for TokenizerCompressionFilterStep {
    fn name(&self) -> &'static str {
        "TokenizerCompressionFilter"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        let tokenizer = self.tokenizer();

        batch
            .into_par_iter()
            .map(|data| {
                if self.should_filter(data.as_ref(), &tokenizer) {
                    StepOutcome::Exclude(data)
                } else {
                    StepOutcome::Keep(data)
                }
            })
            .collect()
    }
}
