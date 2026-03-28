use axum::{
    Json,
    extract::State,
    http::{Request, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Debug, Default)]
pub struct AdminAuthConfig {
    pub api_key: Option<String>,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AuthError {
    #[error("admin api key is not configured")]
    NotConfigured,
    #[error("missing Authorization header")]
    MissingAuthorizationHeader,
    #[error("invalid Authorization header")]
    InvalidAuthorizationHeader,
    #[error("invalid api key")]
    InvalidApiKey,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    message: String,
}

pub async fn auth(
    State(cfg): State<AdminAuthConfig>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let token = match request.headers().get(AUTHORIZATION) {
        Some(value) => value
            .to_str()
            .map(Some)
            .map_err(|_| AuthError::InvalidAuthorizationHeader),
        None => Ok(None),
    };

    match token.and_then(|token| check_api_key(token, &cfg)) {
        Ok(()) => next.run(request).await,
        Err(err) => (
            match err {
                AuthError::NotConfigured => StatusCode::FORBIDDEN,
                AuthError::MissingAuthorizationHeader
                | AuthError::InvalidAuthorizationHeader
                | AuthError::InvalidApiKey => StatusCode::UNAUTHORIZED,
            },
            Json(ErrorResponse {
                message: err.to_string(),
            }),
        )
            .into_response(),
    }
}

pub fn check_api_key(token: Option<&str>, cfg: &AdminAuthConfig) -> Result<(), AuthError> {
    let Some(expected) = cfg
        .api_key
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return Err(AuthError::NotConfigured);
    };

    let Some(token) = token else {
        return Err(AuthError::MissingAuthorizationHeader);
    };

    let token = token.strip_prefix("Bearer ").unwrap_or(token);
    if token == expected {
        Ok(())
    } else {
        Err(AuthError::InvalidApiKey)
    }
}
