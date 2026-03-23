//! 路由模块负责组织对外暴露的推理入口，并把请求接入对应处理链。

pub mod admin;
pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod images;
pub mod models;
pub mod responses;
//
// use axum::{
//     Json,
//     Router,
//     extract::{DefaultBodyLimit, State},
//     http::{Method, StatusCode},
//     routing::{get, post},
// };
// use tower_http::cors::{AllowOrigin, CorsLayer};
//
// use rwkv_config::validated::infer::INFER_CFG;
//
// use crate::routes::chat::completions::chat_completions;
//
// #[cfg(feature = "trace")]
// use tower_http::trace::TraceLayer;
//
//
// #[derive(Clone)]
// pub struct AppState {}
//
// pub struct HttpApiRouterBuilder {
//     app_state: AppState,
// }
//
// impl HttpApiRouterBuilder {
//     pub fn new(app_state: AppState) -> Self {
//         Self { app_state }
//     }
//
//     pub async fn build(self) -> Router {
//         let allow_origin = if let Some(origins) = INFER_CFG.get().unwrap().allowed_origins {
//             let parsed: Result<Vec<_>, _> =
//                 origins.into_iter().map(|origin| origin.parse()).collect();
//             match parsed {
//                 Ok(origins) => AllowOrigin::list(origins),
//                 Err(_) => {
//                     return panic!("invalid allowed origin format");
//                 }
//             }
//         } else {
//             AllowOrigin::any()
//         };
//
//         let cors_layer = CorsLayer::new()
//             .allow_methods([Method::GET, Method::POST])
//             .allow_headers([
//                 axum::http::header::CONTENT_TYPE,
//                 axum::http::header::AUTHORIZATION,
//             ])
//             .allow_origin(allow_origin);
//
//         let router = Router::new()
//             .route("/v1/chat/completions", post(chat_completions))
//             .route("/v1/completions", post(completions))
//             .route("/v1/embeddings", post(embeddings))
//             .route("/v1/responses", post(responses_create))
//             .route(
//                 "/v1/responses/{response_id}",
//                 get(responses_get).delete(responses_delete),
//             )
//             .route("/v1/responses/{response_id}/cancel", post(responses_cancel))
//             .route("/v1/models", get(models))
//             .route("/admin/models/reload", post(admin_models_reload))
//             .route("/v1/images/generations", post(images_generations))
//             .route("/v1/audio/speech", post(audio_speech))
//             .route("/health", get(health))
//             .layer(cors_layer)
//             .layer(DefaultBodyLimit::max(
//                 INFER_CFG.get().unwrap().request_body_limit_bytes,
//             ))
//             .with_state(self.app_state);
//
//         #[cfg(feature = "trace")]
//         let router = router.layer(
//             TraceLayer::new_for_http()
//                 .make_span_with(|request: &axum::http::Request<_>| {
//                     tracing::info_span!(
//                         "rwkv.infer.http.request",
//                         method = %request.method(),
//                         path = %request.uri().path()
//                     )
//                 })
//                 .on_request(|request: &axum::http::Request<_>, _span: &tracing::Span| {
//                     tracing::trace!(
//                         method = %request.method(),
//                         path = %request.uri().path(),
//                         "http request received"
//                     );
//                 })
//                 .on_response(
//                     |response: &axum::http::Response<_>,
//                      latency: std::time::Duration,
//                      _span: &tracing::Span| {
//                         tracing::info!(
//                             status = response.status().as_u16(),
//                             latency_ms = latency.as_millis() as u64,
//                             "http response"
//                         );
//                     },
//                 ),
//         );
//
//         router
//     }
// }
//
// async fn health() -> (StatusCode, &'static str) {
//     (StatusCode::OK, "ok")
// }
//
// async fn models(State(app_state): State<AppState>) -> Json<ModelListResponse> {
//     let api = HttpApiService::new(app_state.runtime_manager.clone());
//     Json(api.models())
// }
