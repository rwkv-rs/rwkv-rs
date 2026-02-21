use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use tower_http::cors::{AllowOrigin, CorsLayer};
#[cfg(feature = "trace")]
use tower_http::trace::TraceLayer;

use crate::api::ApiService;
use crate::auth::AuthConfig;
use crate::server::handlers;
use crate::server::openai_types::ModelListResponse;
use crate::service::RuntimeManager;

#[derive(Clone)]
pub struct AppState {
    pub auth_cfg: AuthConfig,
    pub runtime_manager: Arc<RuntimeManager>,
}

pub struct RouterBuilder {
    app_state: AppState,
}

impl RouterBuilder {
    pub fn new(app_state: AppState) -> Self {
        Self { app_state }
    }

    pub async fn build(self) -> crate::Result<Router> {
        let allow_origin = if let Some(origins) = self.app_state.runtime_manager.allowed_origins() {
            let parsed: Result<Vec<_>, _> =
                origins.into_iter().map(|origin| origin.parse()).collect();
            match parsed {
                Ok(origins) => AllowOrigin::list(origins),
                Err(_) => {
                    return Err(crate::Error::bad_request("invalid allowed origin format"));
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
                "/v1/responses/{response_id}",
                get(handlers::responses_get).delete(handlers::responses_delete),
            )
            .route(
                "/v1/responses/{response_id}/cancel",
                post(handlers::responses_cancel),
            )
            .route("/v1/models", get(models))
            .route("/admin/models/reload", post(handlers::admin_models_reload))
            .route("/v1/images/generations", post(handlers::images_generations))
            .route("/v1/audio/speech", post(handlers::audio_speech))
            .route("/health", get(health))
            .layer(cors_layer)
            .layer(DefaultBodyLimit::max(
                self.app_state.runtime_manager.request_body_limit_bytes(),
            ))
            .with_state(self.app_state);

        #[cfg(feature = "trace")]
        let router = router.layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    tracing::info_span!(
                        "rwkv.infer.http.request",
                        method = %request.method(),
                        path = %request.uri().path()
                    )
                })
                .on_request(|request: &axum::http::Request<_>, _span: &tracing::Span| {
                    tracing::trace!(
                        method = %request.method(),
                        path = %request.uri().path(),
                        "http request received"
                    );
                })
                .on_response(
                    |response: &axum::http::Response<_>,
                     latency: std::time::Duration,
                     _span: &tracing::Span| {
                        tracing::info!(
                            status = response.status().as_u16(),
                            latency_ms = latency.as_millis() as u64,
                            "http response"
                        );
                    },
                ),
        );

        Ok(router)
    }
}

async fn health() -> (StatusCode, &'static str) {
    (StatusCode::OK, "ok")
}

async fn models(State(app_state): State<AppState>) -> Json<ModelListResponse> {
    let api = ApiService::new(app_state.runtime_manager.clone());
    Json(api.models())
}
