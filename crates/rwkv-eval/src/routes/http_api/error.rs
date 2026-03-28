use axum::{
    Json,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::services::{ServiceError, ServiceErrorKind};

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub message: String,
}

#[derive(Debug)]
pub struct ApiError {
    status: axum::http::StatusCode,
    message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                message: self.message,
            }),
        )
            .into_response()
    }
}

impl From<ServiceError> for ApiError {
    fn from(value: ServiceError) -> Self {
        match value.kind() {
            ServiceErrorKind::BadRequest => Self::bad_request(value.message()),
            ServiceErrorKind::Conflict => Self::conflict(value.message()),
            ServiceErrorKind::Forbidden => Self::forbidden(value.message()),
            ServiceErrorKind::Internal => Self::internal(value.message()),
            ServiceErrorKind::NotFound => Self::not_found(value.message()),
        }
    }
}

pub type ApiResult<T> = Result<Json<T>, ApiError>;
