use crate::types::SamplingConfig;

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
    pub model_name: String,
    pub input_text: String,
    pub sampling: SamplingConfig,
    pub stop_suffixes: Vec<String>,
    pub stream: bool,
}
