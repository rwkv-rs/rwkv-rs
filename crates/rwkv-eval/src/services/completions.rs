use crate::{
    db::{CompletionDetailRecord, Db, get_completion_detail},
    services::{ServiceError, ServiceResult},
};

pub async fn detail(db: &Db, completions_id: i32) -> ServiceResult<Option<CompletionDetailRecord>> {
    get_completion_detail(db, completions_id)
        .await
        .map_err(ServiceError::internal)
}
