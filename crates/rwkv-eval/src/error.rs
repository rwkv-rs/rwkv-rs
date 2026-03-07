use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("eval config error: {0}")]
    Config(String),
    #[error("benchmark error: {0}")]
    Benchmark(#[from] BenchmarkError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("openai request error: {0}")]
    OpenAi(#[from] async_openai::error::OpenAIError),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing dataset directory: {0}")]
    MissingDatasetDir(String),
    #[error("missing dataset file: {0}")]
    MissingDatasetFile(String),
    #[error("unsupported benchmark field: {0}")]
    UnsupportedField(String),
    #[error("unsupported benchmark name: {0}")]
    UnsupportedBenchmark(String),
    #[error("dataset validation failed: {0}")]
    InvalidDataset(String),
}
