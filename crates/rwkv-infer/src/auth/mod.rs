use axum::http::header::AUTHORIZATION;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::server::OpenAiErrorResponse;

#[derive(Clone, Debug, Default)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthError {
    MissingAuthorizationHeader,
    InvalidAuthorizationHeader,
    InvalidApiKey,
}

impl AuthError {
    pub fn message(self) -> &'static str {
        match self {
            Self::MissingAuthorizationHeader => "missing Authorization header",
            Self::InvalidAuthorizationHeader => "invalid Authorization header",
            Self::InvalidApiKey => "invalid api key",
        }
    }
}

pub fn check_api_key(headers: &HeaderMap, cfg: &AuthConfig) -> Result<(), Response> {
    let token = match headers.get(AUTHORIZATION) {
        Some(value) => match value.to_str() {
            Ok(token) => Some(token),
            Err(_) => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    axum::Json(OpenAiErrorResponse::unauthorized(
                        AuthError::InvalidAuthorizationHeader.message(),
                    )),
                )
                    .into_response());
            }
        },
        None => None,
    };

    check_api_key_token(token, cfg).map_err(|err| {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(OpenAiErrorResponse::unauthorized(err.message())),
        )
            .into_response()
    })
}

pub fn check_api_key_token(token: Option<&str>, cfg: &AuthConfig) -> Result<(), AuthError> {
    let Some(expected) = cfg.api_key.as_deref() else {
        return Ok(());
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
