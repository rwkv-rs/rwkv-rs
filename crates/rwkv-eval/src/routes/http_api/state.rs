use std::{sync::Arc, time::Duration};

use reqwest::Client;
use rwkv_config::raw::eval::RawEvalConfig;

use crate::{db::Db, handlers::auth::AdminAuthConfig, services::admin::eval::EvalController};

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
}
