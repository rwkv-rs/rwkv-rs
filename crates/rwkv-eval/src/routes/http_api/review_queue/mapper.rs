use crate::{
    db::ReviewQueueRecord,
    dtos::ReviewQueueResource,
    routes::http_api::{
        error::ApiError,
        tasks::{to_checker_summary, to_task_resource},
    },
};

pub(crate) fn to_review_queue_resource(
    record: ReviewQueueRecord,
) -> Result<ReviewQueueResource, ApiError> {
    Ok(ReviewQueueResource {
        task: to_task_resource(&record.task)?,
        completions_id: record.completions_id,
        sample_index: record.sample_index,
        avg_repeat_index: record.avg_repeat_index,
        pass_index: record.pass_index,
        answer: record.answer,
        ref_answer: record.ref_answer,
        fail_reason: record.fail_reason,
        context_preview: record.context_preview,
        checker: to_checker_summary(&record.checker),
    })
}
