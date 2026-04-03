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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs, sync::Arc};

    use axum::{
        body::{Body, to_bytes},
        http::{
            Method,
            Request,
            StatusCode,
            header::{
                ACCESS_CONTROL_ALLOW_METHODS,
                ACCESS_CONTROL_REQUEST_METHOD,
                CONTENT_TYPE,
                ORIGIN,
            },
        },
    };
    use rwkv_data::tokenizer::Tokenizer;
    use tower::ServiceExt;
    use uuid::Uuid;

    use super::*;
    use crate::{
        cores::{
            forward::{ModelForward, StepMode, TokenId},
            queue::queue_worker::{QueueHandle, spawn_queue_worker},
        },
        dtos::errors::OpenAiErrorResponse,
    };

    const TEST_MODEL: &str = "test-model";

    struct DummyModelForward;

    impl ModelForward for DummyModelForward {
        fn step(
            &mut self,
            batch_ids: &[usize],
            _contexts: &[&[i32]],
            _masks: &[&[u8]],
            mode: StepMode<'_>,
        ) -> Option<Vec<TokenId>> {
            match mode {
                StepMode::PrefillNoOutput => None,
                StepMode::Sample { .. } => Some(
                    batch_ids
                        .iter()
                        .copied()
                        .map(|batch_index| TokenId {
                            batch_index,
                            token_id: 1,
                            logprob: None,
                            finish_after_token: true,
                        })
                        .collect(),
                ),
            }
        }

        fn reset(&mut self, _batch_index: usize) {}
    }

    fn test_tokenizer() -> Arc<Tokenizer> {
        let vocab_path = std::env::temp_dir().join(format!("rwkv-infer-{}.txt", Uuid::new_v4()));
        fs::write(&vocab_path, "1 \"a\" 1\n2 \"b\" 1\n").expect("write test vocab");
        Arc::new(Tokenizer::new(vocab_path.to_str().expect("vocab path")).expect("tokenizer"))
    }

    fn test_queue_handle() -> QueueHandle {
        spawn_queue_worker(
            Box::new(DummyModelForward),
            test_tokenizer(),
            4,
            2,
            0,
            "test-weights".to_string(),
        )
    }

    async fn test_app_with_model() -> (axum::Router, QueueHandle) {
        let handle = test_queue_handle();
        let app = HttpApiRouterBuilder::new(AppState::new(
            AuthConfig::default(),
            1024 * 1024,
            1000,
            None,
            HashMap::from([(TEST_MODEL.to_string(), vec![handle.clone()])]),
        ))
        .build()
        .await;
        (app, handle)
    }

    #[tokio::test]
    async fn delete_preflight_is_allowed_by_cors() {
        let app = HttpApiRouterBuilder::new(AppState::new(
            AuthConfig::default(),
            1024,
            1000,
            None,
            HashMap::new(),
        ))
        .build()
        .await;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/v1/responses/resp_123")
                    .header(ORIGIN, "https://example.com")
                    .header(ACCESS_CONTROL_REQUEST_METHOD, "DELETE")
                    .body(Body::empty())
                    .expect("build request"),
            )
            .await
            .expect("router response");

        assert!(response.status().is_success());
        let allow_methods = response
            .headers()
            .get(ACCESS_CONTROL_ALLOW_METHODS)
            .expect("cors allow methods header")
            .to_str()
            .expect("allow methods header");
        assert!(allow_methods.contains("DELETE"));
    }

    #[tokio::test]
    async fn chat_completions_accepts_tool_parameters_objects() {
        let (app, handle) = test_app_with_model().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(
                        r#"{{
                            "model":"{TEST_MODEL}",
                            "messages":[{{"role":"user","content":"hi"}}],
                            "tools":[{{
                                "type":"function",
                                "function":{{
                                    "name":"exec",
                                    "parameters":{{"type":"object","properties":{{}},"additionalProperties":false}}
                                }}
                            }}],
                            "stop":["\n\nUser: "]
                        }}"#
                    )))
                    .expect("build request"),
            )
            .await
            .expect("router response");

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let error: OpenAiErrorResponse = sonic_rs::from_slice(&body).expect("parse error body");

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            error.error.message,
            "stop is not supported together with tools"
        );

        handle.shutdown().await;
    }

    #[tokio::test]
    async fn chat_completions_accepts_response_format_schema_objects() {
        let (app, handle) = test_app_with_model().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(
                        r#"{{
                            "model":"{TEST_MODEL}",
                            "messages":[{{"role":"user","content":"hi"}}],
                            "response_format":{{
                                "type":"json_schema",
                                "json_schema":{{
                                    "name":"result",
                                    "schema":{{"type":"object","properties":{{}},"additionalProperties":false}}
                                }}
                            }},
                            "stop":"done"
                        }}"#
                    )))
                    .expect("build request"),
            )
            .await
            .expect("router response");

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let error: OpenAiErrorResponse = sonic_rs::from_slice(&body).expect("parse error body");

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            error.error.message,
            "stop is not supported together with response_format"
        );

        handle.shutdown().await;
    }
}
