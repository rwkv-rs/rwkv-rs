use axum::Json;
use rwkv_config::raw::eval::{ExtApiConfig, IntApiConfig, RawEvalConfig, SpaceDbConfig};

use super::super::mapper::to_admin_dependency_resource;
use crate::{
    dtos::{
        AdminEvalConfigDto,
        AdminEvalExtApiConfigDto,
        AdminEvalModelConfigDto,
        AdminEvalSpaceDbConfigDto,
        AdminEvalStatusResponse,
    },
    routes::http_api::{
        AppState,
        error::{ApiError, ApiResult},
        state::validate_space_db_config,
        tasks::to_task_resource,
    },
    services::{admin::EvalRunSnapshot, tasks},
};

pub(crate) async fn admin_eval_status_response(
    state: &AppState,
) -> ApiResult<AdminEvalStatusResponse> {
    let snapshot = state.eval_controller.snapshot().await?;
    Ok(Json(build_admin_eval_status(state, snapshot).await?))
}

pub(crate) fn build_admin_eval_draft(service_cfg: &RawEvalConfig) -> AdminEvalConfigDto {
    let mut cfg = service_cfg.clone();
    cfg.fill_default();
    cfg.admin_api_key = None;
    cfg.upload_to_space = Some(true);
    to_admin_eval_config_dto(cfg)
}

pub(crate) fn parse_admin_eval_request(
    payload: AdminEvalConfigDto,
    service_cfg: &RawEvalConfig,
) -> Result<RawEvalConfig, ApiError> {
    let mut cfg = from_admin_eval_config_dto(payload);
    cfg.fill_default();
    validate_admin_eval_config(&cfg)?;
    validate_space_db_matches_service_config(&cfg, service_cfg)?;
    Ok(cfg)
}

async fn build_admin_eval_status(
    state: &AppState,
    snapshot: Option<EvalRunSnapshot>,
) -> Result<AdminEvalStatusResponse, ApiError> {
    let Some(snapshot) = snapshot else {
        return Ok(AdminEvalStatusResponse {
            status: "idle".to_string(),
            desired_state: None,
            config_path: None,
            started_at_unix_ms: None,
            updated_at_unix_ms: None,
            finished_at_unix_ms: None,
            error: None,
            tasks: Vec::new(),
            tasks_total: 0,
            attempts_planned: 0,
            attempts_completed: 0,
            progress_percent: 0.0,
            dependencies: Vec::new(),
        });
    };

    let rows = tasks::by_config_path(&state.db, snapshot.config_path.clone()).await?;
    let tasks = rows
        .iter()
        .map(to_task_resource)
        .collect::<Result<Vec<_>, _>>()?;
    let attempts_planned = tasks
        .iter()
        .map(|task| task.progress.planned_attempts)
        .sum::<u64>();
    let attempts_completed = tasks
        .iter()
        .map(|task| task.progress.completed_attempts)
        .sum::<u64>();
    let progress_percent = if attempts_planned == 0 {
        0.0
    } else {
        ((attempts_completed.min(attempts_planned)) as f64 / attempts_planned as f64)
            .clamp(0.0, 1.0)
    };
    let status = snapshot.runtime.observed_status.as_str().to_string();
    let started_at_unix_ms = snapshot.runtime.started_at_unix_ms;
    let updated_at_unix_ms = snapshot.runtime.updated_at_unix_ms;
    let finished_at_unix_ms = snapshot.runtime.finished_at_unix_ms;
    let error = snapshot.runtime.error;
    let dependencies = snapshot
        .runtime
        .dependencies
        .into_iter()
        .map(to_admin_dependency_resource)
        .collect();

    Ok(AdminEvalStatusResponse {
        status,
        desired_state: Some(snapshot.desired_state.as_str().to_string()),
        config_path: Some(snapshot.config_path),
        started_at_unix_ms: Some(started_at_unix_ms),
        updated_at_unix_ms: Some(updated_at_unix_ms),
        finished_at_unix_ms,
        error,
        tasks_total: tasks.len() as u64,
        attempts_planned,
        attempts_completed,
        progress_percent,
        tasks,
        dependencies,
    })
}

fn validate_admin_eval_config(cfg: &RawEvalConfig) -> Result<(), ApiError> {
    validate_space_db_config(&cfg.space_db).map_err(ApiError::bad_request)?;

    require_non_empty(
        &cfg.llm_judger.api_key,
        "llm_judger.api_key cannot be empty",
    )?;

    if !cfg.skip_checker.unwrap_or(false) {
        require_non_empty(
            &cfg.llm_checker.api_key,
            "llm_checker.api_key cannot be empty",
        )?;
    }

    Ok(())
}

fn validate_space_db_matches_service_config(
    cfg: &RawEvalConfig,
    service_cfg: &RawEvalConfig,
) -> Result<(), ApiError> {
    let mut expected = service_cfg.clone();
    expected.fill_default();

    if cfg.space_db == expected.space_db {
        return Ok(());
    }

    let mut changed_fields = Vec::new();
    if cfg.space_db.username != expected.space_db.username {
        changed_fields.push("username");
    }
    if cfg.space_db.password != expected.space_db.password {
        changed_fields.push("password");
    }
    if cfg.space_db.host != expected.space_db.host {
        changed_fields.push("host");
    }
    if cfg.space_db.port != expected.space_db.port {
        changed_fields.push("port");
    }
    if cfg.space_db.database_name != expected.space_db.database_name {
        changed_fields.push("database_name");
    }
    if cfg.space_db.sslmode != expected.space_db.sslmode {
        changed_fields.push("sslmode");
    }

    Err(ApiError::bad_request(format!(
        "admin eval start does not allow overriding space_db; changed fields: {}",
        changed_fields.join(", ")
    )))
}

fn require_non_empty(value: &str, message: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(ApiError::bad_request(message));
    }
    Ok(())
}

fn to_admin_eval_config_dto(cfg: RawEvalConfig) -> AdminEvalConfigDto {
    AdminEvalConfigDto {
        experiment_name: cfg.experiment_name,
        experiment_desc: cfg.experiment_desc,
        admin_api_key: cfg.admin_api_key,
        run_mode: cfg.run_mode,
        skip_checker: cfg.skip_checker,
        judger_concurrency: cfg.judger_concurrency,
        checker_concurrency: cfg.checker_concurrency,
        db_pool_max_connections: cfg.db_pool_max_connections,
        model_arch_versions: cfg.model_arch_versions,
        model_data_versions: cfg.model_data_versions,
        model_num_params: cfg.model_num_params,
        benchmark_field: cfg.benchmark_field,
        extra_benchmark_name: cfg.extra_benchmark_name,
        upload_to_space: cfg.upload_to_space,
        git_hash: cfg.git_hash,
        models: cfg
            .models
            .into_iter()
            .map(|model| AdminEvalModelConfigDto {
                model_arch_version: model.model_arch_version,
                model_data_version: model.model_data_version,
                model_num_params: model.model_num_params,
                base_url: model.base_url,
                api_key: model.api_key,
                model: model.model,
                max_batch_size: model.max_batch_size,
            })
            .collect(),
        llm_judger: AdminEvalExtApiConfigDto {
            base_url: cfg.llm_judger.base_url,
            api_key: cfg.llm_judger.api_key,
            model: cfg.llm_judger.model,
        },
        llm_checker: AdminEvalExtApiConfigDto {
            base_url: cfg.llm_checker.base_url,
            api_key: cfg.llm_checker.api_key,
            model: cfg.llm_checker.model,
        },
        space_db: AdminEvalSpaceDbConfigDto {
            username: cfg.space_db.username,
            password: cfg.space_db.password,
            host: cfg.space_db.host,
            port: cfg.space_db.port,
            database_name: cfg.space_db.database_name,
            sslmode: cfg.space_db.sslmode,
        },
    }
}

fn from_admin_eval_config_dto(payload: AdminEvalConfigDto) -> RawEvalConfig {
    RawEvalConfig {
        experiment_name: payload.experiment_name,
        experiment_desc: payload.experiment_desc,
        admin_api_key: payload.admin_api_key,
        run_mode: payload.run_mode,
        skip_checker: payload.skip_checker,
        judger_concurrency: payload.judger_concurrency,
        checker_concurrency: payload.checker_concurrency,
        db_pool_max_connections: payload.db_pool_max_connections,
        model_arch_versions: payload.model_arch_versions,
        model_data_versions: payload.model_data_versions,
        model_num_params: payload.model_num_params,
        benchmark_field: payload.benchmark_field,
        extra_benchmark_name: payload.extra_benchmark_name,
        upload_to_space: payload.upload_to_space,
        git_hash: payload.git_hash,
        models: payload
            .models
            .into_iter()
            .map(|model| IntApiConfig {
                model_arch_version: model.model_arch_version,
                model_data_version: model.model_data_version,
                model_num_params: model.model_num_params,
                base_url: model.base_url,
                api_key: model.api_key,
                model: model.model,
                max_batch_size: model.max_batch_size,
            })
            .collect(),
        llm_judger: ExtApiConfig {
            base_url: payload.llm_judger.base_url,
            api_key: payload.llm_judger.api_key,
            model: payload.llm_judger.model,
        },
        llm_checker: ExtApiConfig {
            base_url: payload.llm_checker.base_url,
            api_key: payload.llm_checker.api_key,
            model: payload.llm_checker.model,
        },
        space_db: SpaceDbConfig {
            username: payload.space_db.username,
            password: payload.space_db.password,
            host: payload.space_db.host,
            port: payload.space_db.port,
            database_name: payload.space_db.database_name,
            sslmode: payload.space_db.sslmode,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{build_admin_eval_draft, from_admin_eval_config_dto, parse_admin_eval_request};
    use crate::dtos::{
        AdminEvalConfigDto,
        AdminEvalExtApiConfigDto,
        AdminEvalModelConfigDto,
        AdminEvalSpaceDbConfigDto,
    };

    fn sample_request() -> AdminEvalConfigDto {
        AdminEvalConfigDto {
            experiment_name: "demo".to_string(),
            experiment_desc: "demo".to_string(),
            admin_api_key: Some("admin-secret".to_string()),
            run_mode: None,
            skip_checker: Some(false),
            judger_concurrency: None,
            checker_concurrency: None,
            db_pool_max_connections: None,
            model_arch_versions: vec!["rwkv7".to_string()],
            model_data_versions: vec!["g1".to_string()],
            model_num_params: vec!["1.5b".to_string()],
            benchmark_field: vec!["Knowledge".to_string()],
            extra_benchmark_name: Vec::new(),
            upload_to_space: Some(false),
            git_hash: "abc".to_string(),
            models: vec![AdminEvalModelConfigDto {
                model_arch_version: "rwkv7".to_string(),
                model_data_version: "g1".to_string(),
                model_num_params: "1.5b".to_string(),
                base_url: "127.0.0.1:8080".to_string(),
                api_key: "model-secret".to_string(),
                model: "demo-model".to_string(),
                max_batch_size: Some(16),
            }],
            llm_judger: AdminEvalExtApiConfigDto {
                base_url: "judge:8080".to_string(),
                api_key: "judger-secret".to_string(),
                model: "judge".to_string(),
            },
            llm_checker: AdminEvalExtApiConfigDto {
                base_url: "checker:8080".to_string(),
                api_key: "checker-secret".to_string(),
                model: "checker".to_string(),
            },
            space_db: AdminEvalSpaceDbConfigDto {
                username: "postgres".to_string(),
                password: "db-secret".to_string(),
                host: "localhost".to_string(),
                port: "5432".to_string(),
                database_name: "rwkv".to_string(),
                sslmode: None,
            },
        }
    }

    #[test]
    fn build_draft_clears_admin_key_and_forces_upload() {
        let service_cfg = from_admin_eval_config_dto(sample_request());
        let draft = build_admin_eval_draft(
            &parse_admin_eval_request(sample_request(), &service_cfg).unwrap(),
        );

        assert_eq!(draft.admin_api_key, None);
        assert_eq!(draft.upload_to_space, Some(true));
        assert_eq!(draft.models[0].api_key, "model-secret");
    }

    #[test]
    fn parse_request_allows_empty_model_api_key() {
        let mut request = sample_request();
        request.models[0].api_key.clear();

        let service_cfg = from_admin_eval_config_dto(sample_request());
        let parsed = parse_admin_eval_request(request, &service_cfg).unwrap();
        assert_eq!(parsed.models[0].api_key, "");
    }

    #[test]
    fn parse_request_rejects_overridden_space_db() {
        let service_cfg = from_admin_eval_config_dto(sample_request());
        let mut request = sample_request();
        request.space_db.host = "other-host".to_string();

        let err = parse_admin_eval_request(request, &service_cfg).unwrap_err();
        assert!(
            format!("{err:?}").contains(
                "admin eval start does not allow overriding space_db; changed fields: host"
            )
        );
    }

    #[test]
    fn parse_request_allows_matching_space_db_after_defaults() {
        let mut service_cfg = from_admin_eval_config_dto(sample_request());
        service_cfg.space_db.sslmode = None;
        let mut request = sample_request();
        request.space_db.sslmode = None;

        let parsed = parse_admin_eval_request(request, &service_cfg).unwrap();
        assert_eq!(parsed.space_db.sslmode.as_deref(), Some("verify-full"));
    }
}
