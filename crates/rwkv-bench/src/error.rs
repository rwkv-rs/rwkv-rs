use std::path::Path;

use plotters::drawing::DrawingAreaErrorKind;

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
    #[error("failed to decode report input: {path}")]
    ReportInputDecode { path: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),
    #[error(transparent)]
    Acquire(#[from] tokio::sync::AcquireError),
    #[error("plot render error: {0}")]
    Plot(String),
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

    pub fn report_input_decode(path: impl AsRef<Path>) -> Self {
        Self::ReportInputDecode {
            path: path.as_ref().display().to_string(),
        }
    }
}

impl<E> From<DrawingAreaErrorKind<E>> for BenchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(value: DrawingAreaErrorKind<E>) -> Self {
        Self::Plot(value.to_string())
    }
}
