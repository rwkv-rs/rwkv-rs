use std::net::SocketAddr;

use axum::Router;
use axum::http::Method;
use axum::routing::get;
use tower_http::cors::{Any, CorsLayer};

use super::handlers::{
    benchmarks, completion_detail, health, index, meta, models, openapi_json, review_queue,
    task_attempts, task_detail, tasks,
};
use super::state::AppState;
use crate::db::Db;

pub fn build_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET])
        .allow_headers(Any);

    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/openapi.json", get(openapi_json))
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
        .layer(cors)
        .with_state(state)
}

pub async fn serve(bind_addr: SocketAddr, db: Db) -> Result<(), String> {
    let router = build_router(AppState::new(db));
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|err| format!("bind {} failed: {err}", bind_addr))?;

    axum::serve(listener, router)
        .await
        .map_err(|err| format!("serve {} failed: {err}", bind_addr))
}
