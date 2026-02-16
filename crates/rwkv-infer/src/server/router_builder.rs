use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use rwkv_config::validated::infer::FinalInferConfig;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::auth::AuthConfig;
use crate::server::handlers;
use crate::server::openai_types::{ModelListResponse, ModelObject};
use crate::service::RwkvInferService;

#[derive(Clone)]
pub struct RwkvInferApp {
    pub cfg: Arc<FinalInferConfig>,
    pub service: Arc<RwkvInferService>,
    pub auth: AuthConfig,
}

pub struct RwkvInferRouterBuilder {
    app: Option<RwkvInferApp>,
}

impl RwkvInferRouterBuilder {
    pub fn new() -> Self {
        Self { app: None }
    }

    pub fn with_app(mut self, app: RwkvInferApp) -> Self {
        self.app = Some(app);
        self
    }

    pub async fn build(self) -> crate::Result<Router> {
        let app = self
            .app
            .ok_or(crate::Error::Internal("app must be set".to_string()))?;

        let allow_origin = if let Some(origins) = app.cfg.allowed_origins.clone() {
            let parsed: Result<Vec<_>, _> = origins.into_iter().map(|origin| origin.parse()).collect();
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
            .layer(DefaultBodyLimit::max(app.cfg.request_body_limit_bytes))
            .with_state(app);

        Ok(router)
    }
}

async fn health() -> (StatusCode, &'static str) {
    (StatusCode::OK, "ok")
}

async fn models(State(app): State<RwkvInferApp>) -> Json<ModelListResponse> {
    let data = app
        .service
        .model_names()
        .into_iter()
        .map(|model_name| ModelObject {
            id: model_name,
            object: "model".to_string(),
            owned_by: "rwkv-rs".to_string(),
        })
        .collect();
    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

