use crate::dtos::admin::models::reload::{ModelsReloadReq, ModelsReloadResp};
use crate::services::{ServiceError, ServiceResult};

pub async fn reload_models(_req: ModelsReloadReq) -> ServiceResult<ModelsReloadResp> {
    Err(ServiceError::not_supported(
        "model reload is not migrated to the new HTTP path yet",
    ))
}
