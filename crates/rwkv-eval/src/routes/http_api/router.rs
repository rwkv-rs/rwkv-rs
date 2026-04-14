use axum::{
    Router,
    http::Method,
    middleware,
    routing::{get, post},
};
use tower_http::cors::{Any, CorsLayer};

use super::{
    admin::{
        admin_eval_cancel,
        admin_eval_draft,
        admin_eval_pause,
        admin_eval_resume,
        admin_eval_start,
        admin_eval_status,
        admin_health,
    },
    completions::completion_detail,
    meta::{benchmarks, meta, models},
    review_queue::review_queue,
    state::AppState,
    system::{health, index, openapi_json},
    tasks::{task_attempts, task_detail, tasks},
};
use crate::handlers::auth::auth;

pub struct HttpApiRouterBuilder {
    app_state: AppState,
}

impl HttpApiRouterBuilder {
    pub fn new(app_state: AppState) -> Self {
        Self { app_state }
    }

    pub fn build(self) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers(Any);

        let admin_routes = Router::new()
            .route("/api/v1/admin/eval/draft", get(admin_eval_draft))
            .route("/api/v1/admin/eval/start", post(admin_eval_start))
            .route("/api/v1/admin/eval/pause", post(admin_eval_pause))
            .route("/api/v1/admin/eval/resume", post(admin_eval_resume))
            .route("/api/v1/admin/eval/cancel", post(admin_eval_cancel))
            .route("/api/v1/admin/eval/status", get(admin_eval_status))
            .route("/api/v1/admin/health", get(admin_health))
            .route_layer(middleware::from_fn_with_state(
                self.app_state.admin_auth_cfg.clone(),
                auth,
            ));

        Router::new()
            .route("/api", get(index))
            .route("/api/health", get(health))
            .route("/api/openapi.json", get(openapi_json))
            .route("/api/v1/meta", get(meta))
            .route("/api/v1/models", get(models))
            .route("/api/v1/benchmarks", get(benchmarks))
            .route("/api/v1/tasks", get(tasks))
            .route("/api/v1/tasks/{task_id}", get(task_detail))
            .route("/api/v1/tasks/{task_id}/attempts", get(task_attempts))
            .route(
                "/api/v1/completions/{completions_id}",
                get(completion_detail),
            )
            .route("/api/v1/review-queue", get(review_queue))
            .merge(admin_routes)
            .layer(cors)
            .with_state(self.app_state)
    }
}

pub fn build_router(state: AppState) -> Router {
    HttpApiRouterBuilder::new(state).build()
}
