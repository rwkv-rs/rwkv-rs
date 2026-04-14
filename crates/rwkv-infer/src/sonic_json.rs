use std::ops::{Deref, DerefMut};

use axum::{
    body::Bytes,
    extract::{FromRequest, Request, rejection::BytesRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
};
use serde::{Serialize, de::DeserializeOwned};

use crate::dtos::errors::OpenAiErrorResponse;

#[derive(Debug, Clone, Copy, Default)]
pub struct SonicJson<T>(pub T);

impl<T> SonicJson<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> SonicJson<T>
where
    T: DeserializeOwned,
{
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SonicJsonRejection> {
        sonic_rs::from_slice(bytes)
            .map(Self)
            .map_err(SonicJsonRejection::Deserialize)
    }
}

impl<T> From<T> for SonicJson<T> {
    fn from(inner: T) -> Self {
        Self(inner)
    }
}

impl<T> Deref for SonicJson<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for SonicJson<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
pub enum SonicJsonRejection {
    MissingJsonContentType,
    Bytes(BytesRejection),
    Deserialize(sonic_rs::Error),
}

impl IntoResponse for SonicJsonRejection {
    fn into_response(self) -> Response {
        match self {
            Self::MissingJsonContentType => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                SonicJson(OpenAiErrorResponse::bad_request(
                    "Expected request with `Content-Type: application/json`",
                )),
            )
                .into_response(),
            Self::Bytes(err) => (
                err.status(),
                SonicJson(OpenAiErrorResponse::bad_request(err.body_text())),
            )
                .into_response(),
            Self::Deserialize(err) => (
                StatusCode::BAD_REQUEST,
                SonicJson(OpenAiErrorResponse::bad_request(format!(
                    "Failed to deserialize the JSON body into the target type: {err}"
                ))),
            )
                .into_response(),
        }
    }
}

impl<T, S> FromRequest<S> for SonicJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = SonicJsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        if !json_content_type(req.headers()) {
            return Err(SonicJsonRejection::MissingJsonContentType);
        }

        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(SonicJsonRejection::Bytes)?;
        Self::from_bytes(&bytes)
    }
}

impl<T> IntoResponse for SonicJson<T>
where
    T: Serialize,
{
    fn into_response(self) -> Response {
        match sonic_rs::to_vec(&self.0) {
            Ok(bytes) => (
                [(CONTENT_TYPE, HeaderValue::from_static("application/json"))],
                bytes,
            )
                .into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(
                    CONTENT_TYPE,
                    HeaderValue::from_static("text/plain; charset=utf-8"),
                )],
                err.to_string(),
            )
                .into_response(),
        }
    }
}

fn json_content_type(headers: &HeaderMap<HeaderValue>) -> bool {
    let Some(content_type) = headers.get(CONTENT_TYPE) else {
        return false;
    };

    let Ok(content_type) = content_type.to_str() else {
        return false;
    };

    let media_type = content_type
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or_default();
    let Some((ty, subtype)) = media_type.split_once('/') else {
        return false;
    };

    ty.eq_ignore_ascii_case("application")
        && (subtype.eq_ignore_ascii_case("json") || subtype.to_ascii_lowercase().ends_with("+json"))
}
