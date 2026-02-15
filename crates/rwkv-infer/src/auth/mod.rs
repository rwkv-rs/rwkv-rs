use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::server::OpenAiErrorResponse;

#[derive(Clone, Debug, Default)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

pub fn check_api_key(headers: &HeaderMap, cfg: &AuthConfig) -> std::result::Result<(), Response> {
    let Some(expected) = cfg.api_key.as_deref() else {
        return Ok(());
    };

    let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) else {
        return Err((
            StatusCode::UNAUTHORIZED,
            axum::Json(OpenAiErrorResponse::unauthorized(
                "missing Authorization header",
            )),
        )
            .into_response());
    };

    let Ok(auth) = auth.to_str() else {
        return Err((
            StatusCode::UNAUTHORIZED,
            axum::Json(OpenAiErrorResponse::unauthorized(
                "invalid Authorization header",
            )),
        )
            .into_response());
    };

    // Accept: "Bearer <key>"
    let token = auth.strip_prefix("Bearer ").unwrap_or(auth);
    if token == expected {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            axum::Json(OpenAiErrorResponse::unauthorized("invalid api key")),
        )
            .into_response())
    }
}
