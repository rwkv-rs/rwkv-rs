use super::{CompletionStatus, TaskStatus};

pub struct ModelInsert {
    pub data_version: String,
    pub arch_version: String,
    pub num_params: String,
    pub model_name: String,
}

pub struct BenchmarkInsert {
    pub benchmark_name: String,
    pub benchmark_split: String,
    pub url: Option<String>,
    pub status: String,
    pub num_samples: i32,
}

pub struct TaskInsert {
    pub config_path: Option<String>,
    pub evaluator: String,
    pub is_param_search: bool,
    pub is_tmp: bool,
    pub status: TaskStatus,
    pub git_hash: String,
    pub model_id: i32,
    pub benchmark_id: i32,
    pub desc: Option<String>,
    pub sampling_config_json: String,
    pub log_path: String,
}

pub struct CompletionInsert {
    pub task_id: i32,
    pub context: String,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub status: CompletionStatus,
}

pub struct EvalInsert {
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
}

pub struct CheckerInsert {
    pub completions_id: i32,
    pub answer_correct: bool,
    pub instruction_following_error: bool,
    pub world_knowledge_error: bool,
    pub math_error: bool,
    pub reasoning_logic_error: bool,
    pub thought_contains_correct_answer: bool,
    pub needs_human_review: bool,
    pub reason: String,
}

pub struct ScoreInsert {
    pub task_id: i32,
    pub cot_mode: String,
    pub metrics_json: String,
}
