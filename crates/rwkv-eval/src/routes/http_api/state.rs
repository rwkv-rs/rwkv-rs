use std::{sync::Arc, time::Duration};

use reqwest::Client;
use rwkv_config::raw::eval::{RawEvalConfig, SpaceDbConfig};

use crate::{
    db::{Db, connect},
    handlers::auth::AdminAuthConfig,
    services::admin::EvalController,
};

#[derive(Clone)]
pub struct AppState {
    pub(crate) db: Db,
    pub(crate) service_config: RawEvalConfig,
    pub(crate) admin_auth_cfg: AdminAuthConfig,
    pub(crate) eval_controller: Arc<EvalController>,
    pub(crate) health_client: Client,
}

impl AppState {
    pub fn new(db: Db, service_config: RawEvalConfig) -> Self {
        Self {
            db,
            admin_auth_cfg: AdminAuthConfig {
                api_key: service_config.admin_api_key.clone(),
            },
            service_config,
            eval_controller: Arc::new(EvalController::new()),
            health_client: Client::builder()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap_or_else(|err| panic!("build health client failed: {err}")),
        }
    }

    pub async fn from_config(
        mut raw_eval_cfg: RawEvalConfig,
        db_pool_max_connections: Option<u32>,
    ) -> Result<Self, String> {
        raw_eval_cfg.fill_default();
        validate_space_db_config(&raw_eval_cfg.space_db)?;

        let max_connections = db_pool_max_connections
            .or(raw_eval_cfg.db_pool_max_connections)
            .unwrap_or(32);
        let db = connect(&raw_eval_cfg.space_db, max_connections).await?;

        Ok(Self::new(db, raw_eval_cfg))
    }
}

pub(crate) fn validate_space_db_config(cfg: &SpaceDbConfig) -> Result<(), String> {
    if cfg.host.trim().is_empty() {
        return Err("space_db.host cannot be empty".to_string());
    }
    if cfg.username.trim().is_empty() {
        return Err("space_db.username cannot be empty".to_string());
    }
    if cfg.password.trim().is_empty() {
        return Err("space_db.password cannot be empty".to_string());
    }
    if cfg.port.trim().is_empty() {
        return Err("space_db.port cannot be empty".to_string());
    }
    if cfg.database_name.trim().is_empty() {
        return Err("space_db.database_name cannot be empty".to_string());
    }

    Ok(())
}
