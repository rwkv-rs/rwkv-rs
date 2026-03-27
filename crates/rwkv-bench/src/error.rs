pub type Result<T> = std::result::Result<T, BenchError>;

#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("external command failed (exit={exit_code:?}): {command}")]
    CommandFailed {
        command: String,
        exit_code: Option<i32>,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Json(#[from] sonic_rs::Error),
    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),
    #[error(transparent)]
    Acquire(#[from] tokio::sync::AcquireError),
}

impl BenchError {
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    pub fn command_failed(command: impl Into<String>, exit_code: Option<i32>) -> Self {
        Self::CommandFailed {
            command: command.into(),
            exit_code,
        }
    }
}
