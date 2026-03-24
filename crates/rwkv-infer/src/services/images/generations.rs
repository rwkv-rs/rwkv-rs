use crate::services::{ServiceError, ServiceResult};

pub async fn generations() -> ServiceResult<()> {
    Err(ServiceError::not_supported(
        "image generation is not supported in the new HTTP path yet",
    ))
}
