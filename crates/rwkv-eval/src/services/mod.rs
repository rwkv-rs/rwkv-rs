pub mod admin;
pub mod completions;
pub mod meta;
pub mod review_queue;
pub mod tasks;

#[derive(Clone, Debug)]
pub struct ServiceError {
    kind: ServiceErrorKind,
    message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceErrorKind {
    BadRequest,
    Conflict,
    Forbidden,
    Internal,
    NotFound,
}

pub type ServiceResult<T> = Result<T, ServiceError>;

impl ServiceError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            kind: ServiceErrorKind::BadRequest,
            message: message.into(),
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self {
            kind: ServiceErrorKind::Conflict,
            message: message.into(),
        }
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self {
            kind: ServiceErrorKind::Forbidden,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ServiceErrorKind::Internal,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            kind: ServiceErrorKind::NotFound,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ServiceErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}
