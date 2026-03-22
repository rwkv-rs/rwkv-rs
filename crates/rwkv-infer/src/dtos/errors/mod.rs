use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAiErrorResponse {
    pub error: OpenAiError,
}

impl OpenAiErrorResponse {
    pub fn not_supported(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "not_supported".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "invalid_request_error".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "authentication_error".to_string(),
                param: None,
                code: None,
            },
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                ty: "internal_error".to_string(),
                param: None,
                code: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAiErrorResponse;

    #[test]
    fn serializes_with_nested_error_object() {
        let resp = OpenAiErrorResponse::bad_request("invalid prompt");
        let json = sonic_rs::to_string(&resp).expect("serialize error response");

        assert_eq!(
            json,
            r#"{"error":{"message":"invalid prompt","type":"invalid_request_error","param":null,"code":null}}"#
        );
    }
}
