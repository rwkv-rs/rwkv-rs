use super::{
    catalog::catalog_for_benchmark,
    error::ApiError,
    json::{parse_sampling_summary, parse_score_summary},
    schema::{
        ApiCompletionStatus,
        ApiTaskStatus,
        BenchmarkField,
        BenchmarkResource,
        CheckerSummary,
        CompletionDetailResponse,
        ModelResource,
        ReviewQueueResource,
        TaskAttemptResource,
        TaskResource,
    },
};
use crate::db::{
    BenchmarkRecord,
    CheckerRecord,
    CompletionDetailRecord,
    ModelRecord,
    ReviewQueueRecord,
    TaskAttemptRecord,
    TaskRecord,
};

pub(crate) fn to_model_resource(model: &ModelRecord) -> ModelResource {
    ModelResource {
        model_id: model.model_id,
        model_name: model.model_name.clone(),
        arch_version: model.arch_version.clone(),
        data_version: model.data_version.clone(),
        num_params: model.num_params.clone(),
        model_version: model.model_version.clone(),
    }
}

pub(crate) fn to_benchmark_resource(benchmark: &BenchmarkRecord) -> BenchmarkResource {
    let catalog = catalog_for_benchmark(&benchmark.benchmark_name);

    BenchmarkResource {
        benchmark_id: benchmark.benchmark_id,
        benchmark_name: benchmark.benchmark_name.clone(),
        display_name: catalog
            .map(|entry| entry.display_name.to_string())
            .unwrap_or_else(|| benchmark.benchmark_name.clone()),
        field: catalog
            .map(|entry| entry.field)
            .unwrap_or(BenchmarkField::Unknown),
        benchmark_split: benchmark.benchmark_split.clone(),
        status: benchmark.status.clone(),
        num_samples: benchmark.num_samples,
        url: benchmark.url.clone(),
        supported_cot_modes: catalog
            .map(|entry| entry.supported_cot_modes.clone())
            .unwrap_or_default(),
        supported_n_shots: catalog
            .map(|entry| entry.supported_n_shots.clone())
            .unwrap_or_default(),
        supported_avg_ks: catalog
            .map(|entry| entry.supported_avg_ks.clone())
            .unwrap_or_default(),
        supported_pass_ks: catalog
            .map(|entry| entry.supported_pass_ks.clone())
            .unwrap_or_default(),
    }
}

pub(crate) fn to_checker_summary(checker: &CheckerRecord) -> CheckerSummary {
    CheckerSummary {
        checker_id: checker.checker_id,
        answer_correct: checker.answer_correct,
        instruction_following_error: checker.instruction_following_error,
        world_knowledge_error: checker.world_knowledge_error,
        math_error: checker.math_error,
        reasoning_logic_error: checker.reasoning_logic_error,
        thought_contains_correct_answer: checker.thought_contains_correct_answer,
        needs_human_review: checker.needs_human_review,
        reason: checker.reason.clone(),
        created_at: checker.created_at.clone(),
    }
}

pub(crate) fn to_task_resource(record: &TaskRecord) -> Result<TaskResource, ApiError> {
    Ok(TaskResource {
        task_id: record.task_id,
        created_at: record.task_created_at.clone(),
        status: ApiTaskStatus::from_db(record.task_status),
        config_path: record.config_path.clone(),
        evaluator: record.evaluator.clone(),
        git_hash: record.git_hash.clone(),
        desc: record.task_desc.clone(),
        log_path: record.log_path.clone(),
        is_tmp: record.is_tmp,
        is_param_search: record.is_param_search,
        model: task_model_resource(record),
        benchmark: task_benchmark_resource(record),
        sampling: parse_sampling_summary(&record.sampling_config_json)?,
        score: parse_score_summary(record)?,
    })
}

pub(crate) fn to_task_attempt_resource(record: &TaskAttemptRecord) -> TaskAttemptResource {
    TaskAttemptResource {
        completions_id: record.completions_id,
        task_id: record.task_id,
        sample_index: record.sample_index,
        avg_repeat_index: record.avg_repeat_index,
        pass_index: record.pass_index,
        completion_status: ApiCompletionStatus::from_db(record.completion_status),
        completion_created_at: record.completion_created_at.clone(),
        answer: record.answer.clone(),
        ref_answer: record.ref_answer.clone(),
        is_passed: record.is_passed,
        fail_reason: record.fail_reason.clone(),
        eval_created_at: record.eval_created_at.clone(),
        context_preview: record.context_preview.clone(),
        checker: record.checker.as_ref().map(to_checker_summary),
    }
}

pub(crate) fn to_completion_detail_response(
    record: CompletionDetailRecord,
) -> Result<CompletionDetailResponse, ApiError> {
    Ok(CompletionDetailResponse {
        task: to_task_resource(&record.task)?,
        completions_id: record.completions_id,
        sample_index: record.sample_index,
        avg_repeat_index: record.avg_repeat_index,
        pass_index: record.pass_index,
        completion_status: ApiCompletionStatus::from_db(record.completion_status),
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

fn task_benchmark_resource(record: &TaskRecord) -> BenchmarkResource {
    to_benchmark_resource(&BenchmarkRecord {
        benchmark_id: record.benchmark_id,
        benchmark_name: record.benchmark_name.clone(),
        benchmark_split: record.benchmark_split.clone(),
        url: record.benchmark_url.clone(),
        status: record.benchmark_status.clone(),
        num_samples: record.num_samples,
    })
}

fn task_model_resource(record: &TaskRecord) -> ModelResource {
    ModelResource {
        model_id: record.model_id,
        model_name: record.model_name.clone(),
        arch_version: record.arch_version.clone(),
        data_version: record.data_version.clone(),
        num_params: record.num_params.clone(),
        model_version: format!(
            "{}_{}_{}",
            record.arch_version, record.data_version, record.num_params
        ),
    }
}
