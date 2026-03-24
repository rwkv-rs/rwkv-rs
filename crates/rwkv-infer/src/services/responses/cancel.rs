use crate::services::{ServiceError, ServiceResult};

pub async fn cancel() -> ServiceResult<()> {
    Err(ServiceError::not_supported(
        "responses cancel is not supported in the new HTTP path yet",
    ))
}
