use super::TaskStatus;

pub struct TaskIdentity {
    pub evaluator: String,
    pub git_hash: String,
    pub model_id: i32,
    pub benchmark_id: i32,
    pub sampling_config_json: String,
}

pub struct TaskListQuery {
    pub limit: i64,
    pub offset: i64,
    pub config_path: Option<String>,
    pub latest_only: bool,
    pub status: Option<TaskStatus>,
    pub cot_mode: Option<String>,
    pub evaluator: Option<String>,
    pub git_hash: Option<String>,
    pub model_name: Option<String>,
    pub arch_version: Option<String>,
    pub data_version: Option<String>,
    pub num_params: Option<String>,
    pub benchmark_name: Option<String>,
    pub benchmark_names: Vec<String>,
    pub has_score: Option<bool>,
    pub include_tmp: bool,
    pub include_param_search: bool,
}

pub struct TaskAttemptsQuery {
    pub limit: i64,
    pub offset: i64,
    pub only_failed: bool,
    pub has_checker: Option<bool>,
    pub needs_human_review: Option<bool>,
    pub sample_index: Option<i32>,
}

pub struct ReviewQueueQuery {
    pub limit: i64,
    pub offset: i64,
    pub model_name: Option<String>,
    pub benchmark_name: Option<String>,
    pub benchmark_names: Vec<String>,
    pub task_id: Option<i32>,
}
