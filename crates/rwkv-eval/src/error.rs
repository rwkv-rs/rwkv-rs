use async_openai::error::OpenAIError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("config error: {0}")]
    Config(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error(transparent)]
    OpenAI(#[from] OpenAIError),
}
