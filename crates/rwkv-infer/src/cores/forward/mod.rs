pub mod logprobs;
pub mod sampling;

use crate::cores::forward::sampling::SamplingConfig;

#[derive(Clone, Copy, Debug)]
pub struct GuidedTokenMaskBatchRef<'a> {
    pub token_masks: &'a [i32],
    pub token_mask_words: usize,
}

impl<'a> GuidedTokenMaskBatchRef<'a> {
    pub fn full_batch_size(&self) -> usize {
        if self.token_mask_words == 0 {
            0
        } else {
            self.token_masks.len() / self.token_mask_words
        }
    }

    pub fn row(self, batch_id: usize) -> &'a [i32] {
        let row_start = batch_id
            .checked_mul(self.token_mask_words)
            .expect("guided token mask row offset overflow");
        let row_end = row_start + self.token_mask_words;
        &self.token_masks[row_start..row_end]
    }
}

pub enum StepMode<'a> {
    PrefillNoOutput,
    Sample {
        sampling_configs: &'a [SamplingConfig],
        token_logprobs_configs: &'a [Option<TokenIdLogprobsConfig>],
        guided_token_mask_ref: Option<GuidedTokenMaskBatchRef<'a>>,
    },
}

pub trait ModelForward: Send + 'static {
    fn step(
        &mut self,
        batch_ids: &[usize],
        contexts: &[&[i32]],
        masks: &[&[u8]],
        mode: StepMode<'_>,
    ) -> Option<Vec<TokenId>>;

    fn reset(&mut self, batch_index: usize);
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenTextLogprobsConfig {
    pub top_logprobs: usize,
    pub candidate_token_texts: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenIdLogprobsConfig {
    pub top_logprobs: usize,
    pub candidate_token_ids: Option<Vec<i32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenId {
    pub batch_index: usize,
    pub token_id: i32,
    pub logprob: Option<Logprob>,
    pub finish_after_token: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Logprob {
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TopLogprob {
    pub token_id: i32,
    pub logprob: f32,
}
