pub mod logprobs;
pub mod sampling;

use crate::cores::forward::{
    sampling::SamplingConfig,
};

pub trait ModelForward: Send + 'static {
    fn forward(&mut self, batch_ids: &[usize], contexts: &[&[i32]], masks: &[&[u8]])
    -> Vec<Logits>;

    fn sample(
        &mut self,
        batch_ids: &[usize],
        logits: Vec<Logits>,
        sampling_configs: &[SamplingConfig],
        token_logprobs_configs: &[Option<TokenIdLogprobsConfig>],
    ) -> Vec<TokenId>;

    fn reset(&mut self, batch_index: usize);
}

#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    pub batch_index: usize,
    pub logits: Vec<f32>,
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