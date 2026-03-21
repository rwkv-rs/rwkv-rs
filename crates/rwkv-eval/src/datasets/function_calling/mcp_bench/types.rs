use serde::{Deserialize, Serialize};
use sonic_rs::{Object as Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBenchTaskSpec {
    pub task_id: String,
    pub task_description: String,
    #[serde(default)]
    pub fuzzy_description: String,
    #[serde(default)]
    pub dependency_analysis: String,
    #[serde(default)]
    pub distraction_servers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct McpBenchItem {
    pub task_file: String,
    pub server_name: String,
    pub combination_name: String,
    pub combination_type: String,
    pub servers: Vec<String>,
    pub task: McpBenchTaskSpec,
}

#[derive(Debug, Deserialize)]
pub struct RawTaskFile {
    pub server_tasks: Vec<RawTaskGroup>,
}

#[derive(Debug, Deserialize)]
pub struct RawTaskGroup {
    pub server_name: String,
    pub tasks: Vec<McpBenchTaskSpec>,
    pub servers: Vec<String>,
    pub combination_name: String,
    pub combination_type: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiLikeConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBenchAvailableTool {
    pub name: String,
    pub original_name: String,
    pub server: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedToolCall {
    pub server: String,
    pub tool: String,
    #[serde(default)]
    pub arguments: Map,
}

impl PlannedToolCall {
    pub fn full_name(&self) -> String {
        format!("{}:{}", self.server, self.tool)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningDecision {
    #[serde(default)]
    pub reasoning: String,
    pub should_continue: bool,
    #[serde(default)]
    pub tool_calls: Vec<PlannedToolCall>,
}

#[derive(Debug, Clone)]
pub struct McpBenchStep {
    pub round_num: usize,
    pub cot: String,
    pub decision: PlanningDecision,
    pub executions: Vec<McpBenchExecutionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBenchExecutionResult {
    pub tool: String,
    pub server: String,
    #[serde(default)]
    pub parameters: Map,
    pub round_num: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub planned_layer: Option<usize>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct McpBenchJudgeRequest {
    pub judge_config: OpenAiLikeConfig,
    pub task: String,
    pub final_solution: String,
    pub total_rounds: usize,
    pub available_tools: BTreeMap<String, McpBenchAvailableTool>,
    pub planning_json_compliance: f64,
    pub accumulated_information: String,
    #[serde(default)]
    pub concrete_task_description: String,
    #[serde(default)]
    pub dependency_analysis: String,
    pub execution_results: Vec<McpBenchExecutionResult>,
}

#[derive(Debug, Deserialize)]
pub struct McpBenchJudgeWire {
    pub ok: bool,
    #[serde(default)]
    pub error: String,
    pub evaluation: Option<McpBenchEvaluation>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct McpBenchEvaluation {
    #[serde(default)]
    pub task_completion_score: f64,
    #[serde(default)]
    pub tool_selection_score: f64,
    #[serde(default)]
    pub planning_effectiveness_and_efficiency_score: f64,
    #[serde(default)]
    pub task_fulfillment: f64,
    #[serde(default)]
    pub grounding: f64,
    #[serde(default)]
    pub tool_appropriateness: f64,
    #[serde(default)]
    pub parameter_accuracy: f64,
    #[serde(default)]
    pub dependency_awareness: f64,
    #[serde(default)]
    pub parallelism_and_efficiency: f64,
    #[serde(default)]
    pub input_schema_compliance: Option<f64>,
    #[serde(default)]
    pub valid_tool_name_rate: Option<f64>,
    #[serde(default)]
    pub execution_success_rate: Option<f64>,
    #[serde(default)]
    pub planning_json_compliance: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct McpBenchPreflightWire {
    pub ok: bool,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub total_servers: usize,
    #[serde(default)]
    pub connected_servers: usize,
    #[serde(default)]
    pub servers_with_tools: usize,
    #[serde(default)]
    pub failures: Vec<McpBenchServerFailure>,
    #[serde(default)]
    pub captured_stdout: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpBenchServerFailure {
    #[serde(default)]
    pub server_name: String,
    #[serde(default)]
    pub connection_status: String,
    #[serde(default)]
    pub has_tools: bool,
    #[serde(default)]
    pub error: String,
}

#[derive(Debug, Clone)]
pub struct McpBenchPreflightSummary {
    pub total_servers: usize,
    pub connected_servers: usize,
    pub servers_with_tools: usize,
    pub failures: Vec<McpBenchServerFailure>,
    pub captured_stdout: String,
}
