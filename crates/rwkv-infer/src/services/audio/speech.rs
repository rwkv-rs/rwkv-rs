use crate::services::{ServiceError, ServiceResult};

pub async fn speech() -> ServiceResult<()> {
    Err(ServiceError::not_supported(
        "audio speech is not supported in the new HTTP path yet",
    ))
}
