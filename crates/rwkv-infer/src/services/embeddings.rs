use crate::services::{ServiceError, ServiceResult};

pub async fn embeddings() -> ServiceResult<()> {
    Err(ServiceError::not_supported(
        "embeddings are not supported in the new HTTP path yet",
    ))
}
