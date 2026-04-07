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

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{
            Request,
            StatusCode,
            header::{AUTHORIZATION, CONTENT_TYPE},
        },
    };
    use rwkv_config::raw::eval::{ExtApiConfig, RawEvalConfig, SpaceDbConfig};
    use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
    use sonic_rs::to_string;
    use tower::util::ServiceExt;

    use super::build_router;
    use crate::{
        db::Db,
        dtos::{
            AdminEvalConfigDto,
            AdminEvalExtApiConfigDto,
            AdminEvalModelConfigDto,
            AdminEvalSpaceDbConfigDto,
        },
        routes::http_api::AppState,
    };

    fn test_app_state(admin_api_key: Option<&str>) -> AppState {
        let pool = PgPoolOptions::new().connect_lazy_with(PgConnectOptions::new());
        let db = Db { pool };
        let service_config = RawEvalConfig {
            experiment_name: "test".to_string(),
            experiment_desc: "test".to_string(),
            admin_api_key: admin_api_key.map(ToOwned::to_owned),
            run_mode: None,
            skip_checker: None,
            skip_dataset_check: None,
            judger_concurrency: None,
            checker_concurrency: None,
            db_pool_max_connections: None,
            model_arch_versions: Vec::new(),
            model_data_versions: Vec::new(),
            model_num_params: Vec::new(),
            benchmark_field: Vec::new(),
            extra_benchmark_name: Vec::new(),
            upload_to_space: None,
            git_hash: "test".to_string(),
            models: Vec::new(),
            llm_judger: ExtApiConfig {
                base_url: "http://127.0.0.1".to_string(),
                api_key: String::new(),
                model: "judge".to_string(),
            },
            llm_checker: ExtApiConfig {
                base_url: "http://127.0.0.1".to_string(),
                api_key: String::new(),
                model: "checker".to_string(),
            },
            space_db: SpaceDbConfig {
                username: "postgres".to_string(),
                password: "secret".to_string(),
                host: "localhost".to_string(),
                port: "5432".to_string(),
                database_name: "rwkv".to_string(),
                sslmode: Some("verify-full".to_string()),
            },
        };
        AppState::new(db, service_config)
    }

    fn sample_start_request() -> AdminEvalConfigDto {
        AdminEvalConfigDto {
            experiment_name: "demo".to_string(),
            experiment_desc: "demo".to_string(),
            admin_api_key: None,
            run_mode: Some("new".to_string()),
            skip_checker: Some(false),
            skip_dataset_check: Some(false),
            judger_concurrency: Some(1),
            checker_concurrency: Some(1),
            db_pool_max_connections: Some(1),
            model_arch_versions: vec!["rwkv7".to_string()],
            model_data_versions: vec!["g1".to_string()],
            model_num_params: vec!["1.5b".to_string()],
            benchmark_field: vec!["Knowledge".to_string()],
            extra_benchmark_name: Vec::new(),
            upload_to_space: Some(true),
            git_hash: "test".to_string(),
            models: vec![AdminEvalModelConfigDto {
                model_arch_version: "rwkv7".to_string(),
                model_data_version: "g1".to_string(),
                model_num_params: "1.5b".to_string(),
                base_url: "http://127.0.0.1".to_string(),
                api_key: "secret".to_string(),
                model: "demo-model".to_string(),
                max_batch_size: Some(8),
            }],
            llm_judger: AdminEvalExtApiConfigDto {
                base_url: "http://127.0.0.1".to_string(),
                api_key: "secret".to_string(),
                model: "judge".to_string(),
            },
            llm_checker: AdminEvalExtApiConfigDto {
                base_url: "http://127.0.0.1".to_string(),
                api_key: "secret".to_string(),
                model: "checker".to_string(),
            },
            space_db: AdminEvalSpaceDbConfigDto {
                username: "postgres".to_string(),
                password: "secret".to_string(),
                host: "localhost".to_string(),
                port: "5432".to_string(),
                database_name: "rwkv".to_string(),
                sslmode: Some("verify-full".to_string()),
            },
        }
    }

    #[tokio::test]
    async fn system_routes_stay_available() {
        let app = build_router(test_app_state(None));

        let index = app
            .clone()
            .oneshot(Request::builder().uri("/api").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(index.status(), StatusCode::OK);

        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let openapi = app
            .oneshot(
                Request::builder()
                    .uri("/api/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(openapi.status(), StatusCode::OK);

        let body = to_bytes(openapi.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("/api/v1/tasks"));
        assert!(body.contains("/api/v1/admin/eval/status"));
    }

    #[tokio::test]
    async fn admin_routes_require_auth_when_key_configured() {
        let app = build_router(test_app_state(Some("secret")));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/admin/eval/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn admin_status_returns_idle_when_authorized() {
        let app = build_router(test_app_state(Some("secret")));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/admin/eval/status")
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"status\":\"idle\""));
    }

    #[tokio::test]
    async fn admin_draft_returns_current_config_shape() {
        let app = build_router(test_app_state(Some("secret")));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/admin/eval/draft")
                    .header(AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"experiment_name\":\"test\""));
        assert!(body.contains("\"admin_api_key\":null"));
        assert!(body.contains("\"upload_to_space\":true"));
    }

    #[tokio::test]
    async fn admin_start_does_not_reject_empty_model_api_key() {
        let app = build_router(test_app_state(Some("secret")));
        let mut request = sample_start_request();
        request.models[0].api_key.clear();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/admin/eval/start")
                    .header(AUTHORIZATION, "Bearer secret")
                    .header("content-type", "application/json")
                    .body(Body::from(to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_ne!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(!body.contains("models[0].api_key cannot be empty"));
    }

    #[tokio::test]
    async fn admin_start_rejects_space_db_override() {
        let app = build_router(test_app_state(Some("secret")));
        let mut request = sample_start_request();
        request.space_db.host = "other-host".to_string();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/admin/eval/start")
                    .header(AUTHORIZATION, "Bearer secret")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(to_string(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("does not allow overriding space_db"));
        assert!(body.contains("host"));
    }
}
