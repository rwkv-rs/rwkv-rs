use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::auth::AuthConfig;
use crate::config::BackendConfig;
use crate::engine::EngineHandle;
use crate::server::handlers;
use crate::server::openai_types::{ModelListResponse, ModelObject};

#[derive(Clone)]
pub struct SharedRwkvInferState {
    pub cfg: BackendConfig,
    pub engine: Arc<EngineHandle>,
    pub auth: AuthConfig,
}

pub struct RwkvInferRouterBuilder {
    state: Option<SharedRwkvInferState>,
    allowed_origins: Option<Vec<String>>,
}

impl RwkvInferRouterBuilder {
    pub fn new() -> Self {
        Self {
            state: None,
            allowed_origins: None,
        }
    }

    pub fn with_state(mut self, state: SharedRwkvInferState) -> Self {
        self.state = Some(state);
        self
    }

    pub fn with_allowed_origins(mut self, allowed_origins: Vec<String>) -> Self {
        self.allowed_origins = Some(allowed_origins);
        self
    }

    pub async fn build(self) -> crate::Result<Router> {
        let state = self
            .state
            .ok_or(crate::Error::Internal("state must be set".to_string()))?;

        let allow_origin = if let Some(origins) = self.allowed_origins {
            let parsed: Result<Vec<_>, _> = origins.into_iter().map(|o| o.parse()).collect();
            match parsed {
                Ok(origins) => AllowOrigin::list(origins),
                Err(_) => {
                    return Err(crate::Error::BadRequest(
                        "invalid allowed origin format".to_string(),
                    ));
                }
            }
        } else {
            AllowOrigin::any()
        };

        let cors_layer = CorsLayer::new()
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
            .allow_origin(allow_origin);

        let router = Router::new()
            .route("/v1/chat/completions", post(handlers::chat_completions))
            .route("/v1/completions", post(handlers::completions))
            .route("/v1/embeddings", post(handlers::embeddings))
            .route("/v1/responses", post(handlers::responses_create))
            .route(
                "/v1/responses/:response_id",
                get(handlers::responses_get).delete(handlers::responses_delete),
            )
            .route(
                "/v1/responses/:response_id/cancel",
                post(handlers::responses_cancel),
            )
            .route("/v1/models", get(models))
            .route("/v1/images/generations", post(handlers::images_generations))
            .route("/v1/audio/speech", post(handlers::audio_speech))
            .route("/health", get(health))
            .layer(cors_layer)
            .layer(DefaultBodyLimit::max(state.cfg.request_body_limit_bytes))
            .with_state(state);

        Ok(router)
    }
}

async fn health() -> (StatusCode, &'static str) {
    (StatusCode::OK, "ok")
}

async fn models() -> Json<ModelListResponse> {
    // A minimal static list. This can be extended when model-loading is wired.
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: "rwkv".to_string(),
            object: "model".to_string(),
            owned_by: "rwkv-rs".to_string(),
        }],
    })
}
