//! 鉴权中间件，负责校验请求中的 API key。
//!
//! ```rust,no_run
//! use axum::{Router, middleware, routing::get};
//! use rwkv_infer::handlers::auth::{AuthConfig, auth};
//!
//! let auth_cfg = AuthConfig {
//!     api_key: Some("secret".to_string()),
//! };
//!
//! let _router = Router::new()
//!     .route("/v1/models", get(|| async { "ok" }))
//!     .route_layer(middleware::from_fn_with_state(auth_cfg, auth));
//! ```

use axum::{
    Json,
    extract::State,
    http::{Request, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};
use thiserror::Error;

use crate::dtos::errors::OpenAiErrorResponse;

#[derive(Clone, Debug, Default)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AuthError {
    #[error("missing Authorization header")]
    MissingAuthorizationHeader,
    #[error("invalid Authorization header")]
    InvalidAuthorizationHeader,
    #[error("invalid api key")]
    InvalidApiKey,
}

pub async fn auth(
    State(cfg): State<AuthConfig>,
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
            StatusCode::UNAUTHORIZED,
            Json(OpenAiErrorResponse::unauthorized(err.to_string())),
        )
            .into_response(),
    }
}

pub fn check_api_key(token: Option<&str>, cfg: &AuthConfig) -> Result<(), AuthError> {
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
