use std::collections::BTreeMap;

use rwkv_eval::datasets::{CoTMode, Field};
use serde::{Deserialize, Serialize};
use sonic_rs::Value;
use utoipa::{IntoParams, ToSchema};

use crate::db::{CompletionStatus, TaskStatus};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkField {
    Knowledge,
    Maths,
    Coding,
    InstructionFollowing,
    FunctionCalling,
    Unknown,
}

impl BenchmarkField {
    pub(crate) fn from_rwkv(field: Field) -> Self {
        match field {
            Field::Knowledge => Self::Knowledge,
            Field::Maths => Self::Maths,
            Field::Coding => Self::Coding,
            Field::InstructionFollowing => Self::InstructionFollowing,
            Field::FunctionCalling => Self::FunctionCalling,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiTaskStatus {
    Running,
    Completed,
    Failed,
}

impl ApiTaskStatus {
    pub(crate) fn from_db(status: TaskStatus) -> Self {
        match status {
            TaskStatus::Running => Self::Running,
            TaskStatus::Completed => Self::Completed,
            TaskStatus::Failed => Self::Failed,
        }
    }

    pub(crate) fn into_db(self) -> TaskStatus {
        match self {
            Self::Running => TaskStatus::Running,
            Self::Completed => TaskStatus::Completed,
            Self::Failed => TaskStatus::Failed,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiCompletionStatus {
    Running,
    Completed,
    Failed,
}

impl ApiCompletionStatus {
    pub(crate) fn from_db(status: CompletionStatus) -> Self {
        match status {
            CompletionStatus::Running => Self::Running,
            CompletionStatus::Completed => Self::Completed,
            CompletionStatus::Failed => Self::Failed,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiCotMode {
    NoCot,
    FakeCot,
    Cot,
}

impl ApiCotMode {
    pub(crate) fn from_eval_mode(mode: CoTMode) -> Self {
        match mode {
            CoTMode::NoCoT => Self::NoCot,
            CoTMode::FakeCoT => Self::FakeCot,
            CoTMode::CoT => Self::Cot,
        }
    }

    pub(crate) fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "nocot" | "no_cot" => Some(Self::NoCot),
            "fakecot" | "fake_cot" => Some(Self::FakeCot),
            "cot" => Some(Self::Cot),
            _ => None,
        }
    }

    pub(crate) fn db_name(self) -> &'static str {
        match self {
            Self::NoCot => "NoCoT",
            Self::FakeCot => "FakeCoT",
            Self::Cot => "CoT",
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct IndexResponse {
    pub service: &'static str,
    pub docs_url: &'static str,
    pub openapi_url: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelResource {
    pub model_id: i32,
    pub model_name: String,
    pub arch_version: String,
    pub data_version: String,
    pub num_params: String,
    pub model_version: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct BenchmarkResource {
    pub benchmark_id: i32,
    pub benchmark_name: String,
    pub display_name: String,
    pub field: BenchmarkField,
    pub benchmark_split: String,
    pub status: String,
    pub num_samples: i32,
    pub url: Option<String>,
    pub supported_cot_modes: Vec<ApiCotMode>,
    pub supported_n_shots: Vec<u8>,
    pub supported_avg_ks: Vec<f32>,
    pub supported_pass_ks: Vec<u8>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SamplingSummary {
    pub cot_mode: Option<ApiCotMode>,
    pub n_shot: Option<u32>,
    pub avg_k: Option<f64>,
    pub pass_ks: Vec<u8>,
    pub judger_model_name: Option<String>,
    pub checker_model_name: Option<String>,
    #[schema(value_type = Object)]
    pub raw: Value,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ScoreSummary {
    pub score_id: i32,
    pub created_at: String,
    pub cot_mode: Option<ApiCotMode>,
    pub passed: Option<u64>,
    pub total: Option<u64>,
    pub sample_size: Option<u64>,
    pub avg_repeat_count: Option<u64>,
    pub max_pass_k: Option<u64>,
    pub pass_at_k: BTreeMap<String, f64>,
    #[schema(value_type = Object)]
    pub metrics: Value,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TaskResource {
    pub task_id: i32,
    pub created_at: String,
    pub status: ApiTaskStatus,
    pub config_path: Option<String>,
    pub evaluator: String,
    pub git_hash: String,
    pub desc: Option<String>,
    pub log_path: String,
    pub is_tmp: bool,
    pub is_param_search: bool,
    pub model: ModelResource,
    pub benchmark: BenchmarkResource,
    pub sampling: SamplingSummary,
    pub score: Option<ScoreSummary>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TaskListResponse {
    pub items: Vec<TaskResource>,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TaskDetailResponse {
    pub task: TaskResource,
    pub attempts_total: u64,
    pub attempts_passed: u64,
    pub attempts_failed: u64,
    pub attempts_with_checker: u64,
    pub attempts_missing_checker: u64,
    pub attempts_needing_human_review: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CheckerSummary {
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

#[derive(Debug, Serialize, ToSchema)]
pub struct TaskAttemptResource {
    pub completions_id: i32,
    pub task_id: i32,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub completion_status: ApiCompletionStatus,
    pub completion_created_at: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
    pub eval_created_at: String,
    pub context_preview: String,
    pub checker: Option<CheckerSummary>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TaskAttemptsResponse {
    pub items: Vec<TaskAttemptResource>,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompletionDetailResponse {
    pub task: TaskResource,
    pub completions_id: i32,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub completion_status: ApiCompletionStatus,
    pub completion_created_at: String,
    pub context: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
    pub eval_created_at: String,
    pub checker: Option<CheckerSummary>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ReviewQueueResource {
    pub task: TaskResource,
    pub completions_id: i32,
    pub sample_index: i32,
    pub avg_repeat_index: i32,
    pub pass_index: i32,
    pub answer: String,
    pub ref_answer: String,
    pub fail_reason: String,
    pub context_preview: String,
    pub checker: CheckerSummary,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ReviewQueueResponse {
    pub items: Vec<ReviewQueueResource>,
    pub limit: u32,
    pub offset: u32,
    pub has_more: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct MetaResponse {
    pub fields: Vec<BenchmarkField>,
    pub task_statuses: Vec<ApiTaskStatus>,
    pub completion_statuses: Vec<ApiCompletionStatus>,
    pub cot_modes: Vec<ApiCotMode>,
    pub models: Vec<ModelResource>,
    pub benchmarks: Vec<BenchmarkResource>,
}

#[derive(Debug, Deserialize, IntoParams)]
pub struct ListTasksParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub latest_only: Option<bool>,
    pub status: Option<ApiTaskStatus>,
    pub cot_mode: Option<ApiCotMode>,
    pub field: Option<BenchmarkField>,
    pub evaluator: Option<String>,
    pub git_hash: Option<String>,
    pub model_name: Option<String>,
    pub arch_version: Option<String>,
    pub data_version: Option<String>,
    pub num_params: Option<String>,
    pub benchmark_name: Option<String>,
    pub has_score: Option<bool>,
    pub include_tmp: Option<bool>,
    pub include_param_search: Option<bool>,
}

#[derive(Debug, Deserialize, IntoParams)]
pub struct TaskAttemptsParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub only_failed: Option<bool>,
    pub has_checker: Option<bool>,
    pub needs_human_review: Option<bool>,
    pub sample_index: Option<i32>,
}

#[derive(Debug, Deserialize, IntoParams)]
pub struct ReviewQueueParams {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub field: Option<BenchmarkField>,
    pub model_name: Option<String>,
    pub benchmark_name: Option<String>,
    pub task_id: Option<i32>,
}
