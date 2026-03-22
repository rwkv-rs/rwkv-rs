pub type Result<T> = std::result::Result<T, Error>;

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not supported: {0}")]
    NotSupported(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("bad request: {context}: {source}")]
    BadRequestWithSource {
        context: String,
        #[source]
        source: BoxError,
    },
    #[error("internal error: {context}: {source}")]
    InternalWithSource {
        context: String,
        #[source]
        source: BoxError,
    },
    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    pub fn not_supported(message: impl Into<String>) -> Self {
        Self::NotSupported(message.into())
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub fn bad_request_with_source(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::BadRequestWithSource {
            context: context.into(),
            source: Box::new(source),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub fn internal_with_source(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::InternalWithSource {
            context: context.into(),
            source: Box::new(source),
        }
    }

    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::NotSupported(_) | Self::BadRequest(_) | Self::BadRequestWithSource { .. }
        )
    }

    pub fn openai_error_type(&self) -> &'static str {
        match self {
            Self::NotSupported(_) => "not_supported",
            Self::BadRequest(_) | Self::BadRequestWithSource { .. } => "invalid_request_error",
            Self::Internal(_) | Self::InternalWithSource { .. } => "internal_error",
        }
    }

    pub fn format_chain(&self) -> String {
        let mut chain = vec![self.to_string()];
        let mut source = std::error::Error::source(self);
        while let Some(err) = source {
            chain.push(err.to_string());
            source = std::error::Error::source(err);
        }
        chain.join(" | caused by: ")
    }
}
