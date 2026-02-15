use crate::config::SamplingConfig;

#[derive(Clone, Debug)]
pub enum InferRequestKind {
    Completion,
    ChatCompletion,
    Embedding,
    Response,
    ImageGeneration,
    Speech,
}

#[derive(Clone, Debug)]
pub struct InferRequest {
    pub kind: InferRequestKind,
    pub input_text: String,
    pub sampling: SamplingConfig,
    pub stream: bool,
}
