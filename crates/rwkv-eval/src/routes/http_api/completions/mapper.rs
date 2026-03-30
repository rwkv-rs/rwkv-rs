use crate::{
    db::CompletionDetailRecord,
    dtos::CompletionDetailResponse,
    routes::http_api::{
        error::ApiError,
        tasks::{to_checker_summary, to_task_resource},
    },
};

pub(crate) fn to_completion_detail_response(
    record: CompletionDetailRecord,
) -> Result<CompletionDetailResponse, ApiError> {
    Ok(CompletionDetailResponse {
        task: to_task_resource(&record.task)?,
        completions_id: record.completions_id,
        sample_index: record.sample_index,
        avg_repeat_index: record.avg_repeat_index,
        pass_index: record.pass_index,
        completion_status: crate::dtos::ApiCompletionStatus::from_db(record.completion_status),
        completion_created_at: record.completion_created_at,
        context: record.context,
        answer: record.answer,
        ref_answer: record.ref_answer,
        is_passed: record.is_passed,
        fail_reason: record.fail_reason,
        eval_created_at: record.eval_created_at,
        checker: record.checker.as_ref().map(to_checker_summary),
    })
}
