#[derive(Clone, Debug, PartialEq)]
pub struct OutputTokenCandidate {
    pub token_id: i32,
    pub logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputToken {
    pub token_id: i32,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<OutputTokenCandidate>,
}
