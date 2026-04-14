pub mod admin;
pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod images;
pub mod models;
pub mod responses;

use std::{path::PathBuf, sync::Arc};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::{delete, get, post},
};
use tower_http::cors::{AllowOrigin, CorsLayer};
#[cfg(feature = "trace")]
use tower_http::trace::TraceLayer;

use crate::{
    handlers::auth::{AuthConfig, auth},
    services::{
        QueueMap,
        QueueMapBuilder,
        SharedQueueMap,
        health::GpuMetricsCache,
        shared_queue_map,
    },
};

#[derive(Clone)]
pub struct AppState {
    pub auth_cfg: AuthConfig,
    pub request_body_limit_bytes: usize,
    pub sse_keep_alive_ms: u64,
    pub allowed_origins: Option<Vec<String>>,
    pub queues: SharedQueueMap,
    pub gpu_metrics: Arc<GpuMetricsCache>,
    pub reload_lock: Option<Arc<tokio::sync::Mutex<()>>>,
    pub infer_cfg_path: Option<PathBuf>,
    pub build_queues: Option<QueueMapBuilder>,
}

impl AppState {
    pub fn new(
        auth_cfg: AuthConfig,
        request_body_limit_bytes: usize,
        sse_keep_alive_ms: u64,
        allowed_origins: Option<Vec<String>>,
        queues: QueueMap,
    ) -> Self {
        Self {
            auth_cfg,
            request_body_limit_bytes,
            sse_keep_alive_ms,
            allowed_origins,
            queues: shared_queue_map(queues),
            gpu_metrics: Arc::new(GpuMetricsCache::new()),
            reload_lock: None,
            infer_cfg_path: None,
            build_queues: None,
        }
    }

    pub fn with_reload_support(
        mut self,
        infer_cfg_path: PathBuf,
        build_queues: QueueMapBuilder,
    ) -> Self {
        self.reload_lock = Some(Arc::new(tokio::sync::Mutex::new(())));
        self.infer_cfg_path = Some(infer_cfg_path);
        self.build_queues = Some(build_queues);
        self
    }
}

pub struct HttpApiRouterBuilder {
    app_state: AppState,
}

impl HttpApiRouterBuilder {
    pub fn new(app_state: AppState) -> Self {
        Self { app_state }
    }

    pub async fn build(self) -> Router {
        let allow_origin = if let Some(origins) = self.app_state.allowed_origins.clone() {
            let parsed: Result<Vec<_>, _> =
                origins.into_iter().map(|origin| origin.parse()).collect();
            AllowOrigin::list(parsed.unwrap())
        } else {
            AllowOrigin::any()
        };

        let cors_layer = CorsLayer::new()
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::DELETE,
            ])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
            .allow_origin(allow_origin);

        let protected = Router::new()
            .route(
                "/v1/chat/completions",
                post(chat::completions::chat_completions),
            )
            .route("/v1/completions", post(completions::completions))
            .route("/v1/embeddings", post(embeddings::embeddings))
            .route("/v1/responses", post(responses::responses_create))
            .route(
                "/v1/responses/{response_id}",
                get(responses::responses_get).route_layer(middleware::from_fn_with_state(
                    self.app_state.auth_cfg.clone(),
                    auth,
                )),
            )
            .route(
                "/v1/responses/{response_id}",
                delete(responses::responses_delete),
            )
            .route(
                "/v1/responses/{response_id}/cancel",
                post(responses::cancel::responses_cancel),
            )
            .route("/v1/models", get(models::models))
            .route(
                "/admin/models/reload",
                post(admin::models::reload::admin_models_reload),
            )
            .route(
                "/v1/images/generations",
                post(images::generations::images_generations),
            )
            .route("/v1/audio/speech", post(audio::speech::audio_speech))
            .route_layer(middleware::from_fn_with_state(
                self.app_state.auth_cfg.clone(),
                auth,
            ));

        let router = Router::new()
            .route("/health", get(health::health))
            .merge(protected)
            .layer(cors_layer)
            .layer(DefaultBodyLimit::max(
                self.app_state.request_body_limit_bytes,
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

        router
    }
}
