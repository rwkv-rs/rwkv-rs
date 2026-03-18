use super::{CompletionStatus, TaskStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskLookup {
    pub task_id: i32,
    pub status: TaskStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompletionKey {
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptRecord {
    pub completions_id: i32,
    pub key: CompletionKey,
    pub context: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub has_checker: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRecord {
    pub model_id: i32,
    pub model_name: String,
    pub arch_version: String,
    pub data_version: String,
    pub num_params: String,
    pub model_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkRecord {
    pub benchmark_id: i32,
    pub benchmark_name: String,
    pub benchmark_split: String,
    pub url: Option<String>,
    pub status: String,
    pub num_samples: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskRecord {
    pub task_id: i32,
    pub config_path: Option<String>,
    pub evaluator: String,
    pub is_param_search: bool,
    pub is_tmp: bool,
    pub task_created_at: String,
    pub task_status: TaskStatus,
    pub git_hash: String,
    pub task_desc: Option<String>,
    pub sampling_config_json: String,
    pub log_path: String,
    pub model_id: i32,
    pub model_name: String,
    pub arch_version: String,
    pub data_version: String,
    pub num_params: String,
    pub benchmark_id: i32,
    pub benchmark_name: String,
    pub benchmark_split: String,
    pub benchmark_url: Option<String>,
    pub benchmark_status: String,
    pub num_samples: i32,
    pub score_id: Option<i32>,
    pub score_created_at: Option<String>,
    pub score_cot_mode: Option<String>,
    pub metrics_json: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskDetailRecord {
    pub task: TaskRecord,
    pub attempts_total: i64,
    pub attempts_passed: i64,
    pub attempts_failed: i64,
    pub attempts_with_checker: i64,
    pub attempts_missing_checker: i64,
    pub attempts_needing_human_review: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckerRecord {
    pub checker_id: i32,
    pub answer_correct: bool,
    pub instruction_following_error: bool,
    pub world_knowledge_error: bool,
    pub math_error: bool,
    pub reasoning_logic_error: bool,
    pub thought_contains_correct_answer: bool,
    pub needs_human_review: bool,
    pub reason: String,
    pub created_at: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskAttemptRecord {
    pub completions_id: i32,
    pub task_id: i32,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub completion_status: CompletionStatus,
    pub completion_created_at: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
    pub eval_created_at: String,
    pub context_preview: String,
    pub checker: Option<CheckerRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionDetailRecord {
    pub task: TaskRecord,
    pub completions_id: i32,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub completion_status: CompletionStatus,
    pub completion_created_at: String,
    pub context: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
    pub eval_created_at: String,
    pub checker: Option<CheckerRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewQueueRecord {
    pub task: TaskRecord,
    pub completions_id: i32,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub answer: String,
    pub ref_answer: String,
    pub fail_reason: String,
    pub context_preview: String,
    pub checker: CheckerRecord,
}
