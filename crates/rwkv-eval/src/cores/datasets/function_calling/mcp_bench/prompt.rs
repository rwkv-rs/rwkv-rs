use super::types::{
    McpBenchAvailableTool, McpBenchExecutionResult, McpBenchItem, McpBenchStep, PlannedToolCall,
    PlanningDecision,
};
use regex::Regex;
use serde::Deserialize;
use sonic_rs::{Object as Map, Value, prelude::*};
use std::collections::BTreeMap;

pub const MCP_BENCH_MAX_ROUNDS: usize = 20;
pub const MCP_BENCH_COT_MAX_TOKENS: u32 = 2048;
pub const MCP_BENCH_DECISION_MAX_TOKENS: u32 = 2048;
pub const MCP_BENCH_FINAL_MAX_TOKENS: u32 = 3072;
pub const MCP_BENCH_MAX_TOOL_SCHEMA_CHARS: usize = 1200;
pub const MCP_BENCH_MAX_RESULT_CHARS: usize = 4000;
pub const MCP_BENCH_MAX_ERROR_CHARS: usize = 1000;
pub const MCP_BENCH_MAX_HISTORY_CHARS: usize = 24000;

pub fn presented_task(item: &McpBenchItem) -> &str {
    let fuzzy = item.task.fuzzy_description.trim();
    if fuzzy.is_empty() {
        item.task.task_description.trim()
    } else {
        fuzzy
    }
}

pub fn build_context_summary(item: &McpBenchItem) -> String {
    format!(
        concat!(
            "benchmark=mcp_bench\n",
            "phase=2_rust_native_completion_loop\n",
            "task_file={}\n",
            "server_name={}\n",
            "servers={}\n",
            "combination_name={}\n",
            "combination_type={}\n",
            "task_id={}\n\n",
            "task_presented_to_agent:\n{}\n\n",
            "concrete_task_reference:\n{}\n\n",
            "dependency_analysis_reference:\n{}\n"
        ),
        item.task_file,
        item.server_name,
        item.servers.join(", "),
        item.combination_name,
        item.combination_type,
        item.task.task_id,
        presented_task(item),
        item.task.task_description,
        item.task.dependency_analysis,
    )
}

pub fn build_planning_context(
    item: &McpBenchItem,
    available_tools: &BTreeMap<String, McpBenchAvailableTool>,
    accumulated_information: &str,
) -> String {
    let history = if accumulated_information.trim().is_empty() {
        "No previous tool results.".to_string()
    } else {
        trim_history(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS)
    };

    format!(
        concat!(
            "System: You are a tool-using MCP benchmark agent operating in a real multi-server environment.\n",
            "You must decide round by round which MCP tools to call.\n",
            "Only plan tool calls that can be executed with information already available at the start of this round.\n",
            "If multiple tool calls are independent, you may include all of them in the same round.\n",
            "Do not invent tool names, parameters, or results.\n",
            "When the task has enough evidence, stop planning and let the final answer be synthesized from gathered evidence.\n\n",
            "User: TASK PRESENTED TO AGENT:\n{}\n\n",
            "AVAILABLE TOOLS:\n{}\n\n",
            "EXECUTION HISTORY:\n{}\n\n",
            "Return reasoning privately inside <think> and then output a strict JSON planning decision.\n",
            "The JSON schema is:\n",
            "{{\n",
            "  \"reasoning\": \"brief planning rationale\",\n",
            "  \"should_continue\": true,\n",
            "  \"tool_calls\": [\n",
            "    {{\"server\": \"Exact Server Name\", \"tool\": \"exact_tool_name\", \"arguments\": {{}}}}\n",
            "  ]\n",
            "}}\n",
            "If no more tools are needed, set \"should_continue\" to false and \"tool_calls\" to [].\n\n",
            "Assistant: <think><|completions_of_cot|>"
        ),
        presented_task(item),
        render_tool_catalog(available_tools),
        history,
    )
}

pub fn build_planning_decision_prompt(cot_context: &str, cot: &str) -> String {
    format!(
        concat!(
            "{}",
            "</think>\n",
            "Return ONLY the JSON planning decision object and nothing else.\n"
        ),
        cot_context.replace("<|completions_of_cot|>", cot)
    )
}

pub fn build_final_answer_prompt(item: &McpBenchItem, accumulated_information: &str) -> String {
    let history = if accumulated_information.trim().is_empty() {
        "No tool evidence was gathered.".to_string()
    } else {
        trim_history(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS)
    };

    format!(
        concat!(
            "System: You are the final answer synthesizer for an MCP benchmark agent.\n",
            "Use only the gathered evidence below. Do not invent missing facts.\n",
            "Return only the final answer requested by the task.\n",
            "If the task requires JSON, return valid JSON and nothing else.\n\n",
            "User: TASK PRESENTED TO AGENT:\n{}\n\n",
            "GATHERED EVIDENCE:\n{}\n\n",
            "Assistant:"
        ),
        presented_task(item),
        history,
    )
}

pub fn append_round_summary(
    accumulated_information: &mut String,
    round_num: usize,
    reasoning: &str,
    executions: &[McpBenchExecutionResult],
) {
    let mut round_summary = format!("\n\n--- Summary of Round {round_num} ---\n");
    if !reasoning.trim().is_empty() {
        round_summary.push_str("Planner reasoning: ");
        round_summary.push_str(reasoning.trim());
        round_summary.push('\n');
    }

    for execution in executions {
        let params = if execution.parameters.is_empty() {
            "{}".to_string()
        } else {
            sonic_rs::to_string(&execution.parameters).unwrap_or_else(|_| "{}".to_string())
        };
        if execution.success {
            let rendered = truncate_text(
                execution.result.as_deref().unwrap_or_default(),
                MCP_BENCH_MAX_RESULT_CHARS,
            );
            round_summary.push_str(&format!(
                "Tool `{}` with Parameter {} on {} succeeded. Result: {}\n",
                execution.tool, params, execution.server, rendered
            ));
        } else {
            let rendered = truncate_text(
                execution.error.as_deref().unwrap_or_default(),
                MCP_BENCH_MAX_ERROR_CHARS,
            );
            round_summary.push_str(&format!(
                "Tool `{}` with Parameter {} on {} failed. Error: {}\n",
                execution.tool, params, execution.server, rendered
            ));
        }
    }

    accumulated_information.push_str(&round_summary);
    if accumulated_information.len() > MCP_BENCH_MAX_HISTORY_CHARS {
        *accumulated_information =
            trim_history(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS);
    }
}

pub fn render_trace(steps: &[McpBenchStep]) -> String {
    let mut text = String::new();
    for step in steps {
        text.push_str(&format!("\n\n[Round {}]\n", step.round_num));
        if !step.cot.trim().is_empty() {
            text.push_str("<think>");
            text.push_str(step.cot.trim());
            text.push_str("</think>\n");
        }
        text.push_str("Decision:\n");
        text.push_str(
            &sonic_rs::to_string_pretty(&step.decision)
                .unwrap_or_else(|_| "{\"error\":\"decision render failed\"}".to_string()),
        );
        text.push('\n');
        for execution in &step.executions {
            text.push_str(&format!(
                "- {} [{}] success={}\n",
                execution.tool, execution.server, execution.success
            ));
        }
    }
    text.trim().to_string()
}

pub fn parse_planning_decision(response: &str) -> Result<PlanningDecision, String> {
    let candidate = extract_json_candidate(response)?;
    if let Ok(decision) = sonic_rs::from_str::<PlanningDecision>(&candidate) {
        if !decision.tool_calls.is_empty() || !candidate.contains("\"planned_tools\"") {
            return Ok(decision);
        }
    }

    let alt = sonic_rs::from_str::<AltPlanningDecision>(&candidate)
        .map_err(|err| format!("failed to parse planning json: {err}; json={candidate}"))?;
    Ok(PlanningDecision {
        reasoning: alt.reasoning.unwrap_or_default().trim().to_string(),
        should_continue: alt.should_continue.unwrap_or(false),
        tool_calls: alt
            .tool_calls
            .into_iter()
            .chain(alt.planned_tools)
            .map(|call| {
                let mut normalized = call.into_planned_tool_call();
                if normalized.server.is_empty() && normalized.tool.contains(':') {
                    if let Some((server, tool)) = normalized.tool.split_once(':') {
                        normalized.server = server.trim().to_string();
                        normalized.tool = tool.trim().to_string();
                    }
                }
                normalized
            })
            .collect(),
    })
}

pub fn normalize_planned_tool_call(
    call: &PlannedToolCall,
    available_tools: &BTreeMap<String, McpBenchAvailableTool>,
) -> Result<PlannedToolCall, String> {
    let mut normalized = call.clone();
    normalized.server = normalized.server.trim().to_string();
    normalized.tool = normalized.tool.trim().to_string();

    if normalized.server.is_empty() && normalized.tool.contains(':') {
        if let Some((server, tool)) = normalized.tool.split_once(':') {
            normalized.server = server.trim().to_string();
            normalized.tool = tool.trim().to_string();
        }
    }

    let full_name = normalized.full_name();
    if available_tools.contains_key(&full_name) {
        return Ok(normalized);
    }

    if normalized.server.is_empty() {
        let mut matches = available_tools
            .keys()
            .filter(|tool_name| tool_name.ends_with(&format!(":{}", normalized.tool)))
            .cloned()
            .collect::<Vec<_>>();
        matches.sort();
        if matches.len() == 1 {
            let matched = matches.pop().unwrap();
            if let Some((server, tool)) = matched.split_once(':') {
                normalized.server = server.to_string();
                normalized.tool = tool.to_string();
                return Ok(normalized);
            }
        }
    }

    Err(format!(
        "planned tool `{}` was not found in available tools",
        if normalized.server.is_empty() {
            normalized.tool.clone()
        } else {
            normalized.full_name()
        }
    ))
}

fn render_tool_catalog(available_tools: &BTreeMap<String, McpBenchAvailableTool>) -> String {
    let mut grouped = BTreeMap::<String, Vec<&McpBenchAvailableTool>>::new();
    for tool in available_tools.values() {
        grouped.entry(tool.server.clone()).or_default().push(tool);
    }

    let mut lines = Vec::new();
    for (server, tools) in grouped {
        lines.push(format!("[{server}]"));
        for tool in tools {
            lines.push(format!(
                "- {}: {}",
                tool.name,
                if tool.description.trim().is_empty() {
                    "No description available".to_string()
                } else {
                    truncate_text(tool.description.trim(), 400)
                }
            ));
            let schema = render_schema_summary(&tool.input_schema);
            if !schema.is_empty() {
                lines.push(format!("  schema: {schema}"));
            }
        }
        lines.push(String::new());
    }
    lines.join("\n").trim().to_string()
}

fn render_schema_summary(schema: &Value) -> String {
    if schema.is_null() {
        return String::new();
    }
    let rendered = sonic_rs::to_string(schema).unwrap_or_default();
    truncate_text(&rendered, MCP_BENCH_MAX_TOOL_SCHEMA_CHARS)
}

fn extract_json_candidate(response: &str) -> Result<String, String> {
    static JSON_BLOCK_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"(?s)```json\s*(?P<body>\{.*?\})\s*```").unwrap());

    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Err("model returned empty planning response".to_string());
    }

    if let Some(caps) = JSON_BLOCK_RE.captures(trimmed) {
        if let Some(body) = caps.name("body") {
            return Ok(body.as_str().trim().to_string());
        }
    }

    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Ok(trimmed.to_string());
    }

    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            return Ok(trimmed[start..=end].trim().to_string());
        }
    }

    Err(format!("model response did not contain json: {trimmed}"))
}

fn trim_history(history: &str, max_chars: usize) -> String {
    if history.len() <= max_chars {
        return history.to_string();
    }
    let keep_tail = max_chars.saturating_sub(64);
    let tail = &history[history.len() - keep_tail..];
    format!("[Earlier execution history truncated]\n\n{tail}")
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..max_chars])
    }
}

#[derive(Debug, Deserialize)]
struct AltPlanningDecision {
    reasoning: Option<String>,
    should_continue: Option<bool>,
    #[serde(default)]
    tool_calls: Vec<AltToolCall>,
    #[serde(default)]
    planned_tools: Vec<AltToolCall>,
}

#[derive(Debug, Deserialize)]
struct AltToolCall {
    server: Option<String>,
    tool: Option<String>,
    #[serde(default)]
    arguments: Map,
    #[serde(default)]
    parameters: Map,
}

impl AltToolCall {
    fn into_planned_tool_call(self) -> PlannedToolCall {
        let arguments = if self.arguments.is_empty() {
            self.parameters
        } else {
            self.arguments
        };
        PlannedToolCall {
            server: self.server.unwrap_or_default().trim().to_string(),
            tool: self.tool.unwrap_or_default().trim().to_string(),
            arguments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{McpBenchAvailableTool, normalize_planned_tool_call, parse_planning_decision};
    use crate::cores::datasets::function_calling::mcp_bench::types::PlannedToolCall;
    use sonic_rs::json;
    use std::collections::BTreeMap;

    #[test]
    fn parses_native_decision_json() {
        let response = r#"{"reasoning":"r","should_continue":true,"tool_calls":[{"server":"S","tool":"t","arguments":{"x":1}}]}"#;
        let decision = parse_planning_decision(response).unwrap();
        assert!(decision.should_continue);
        assert_eq!(decision.tool_calls.len(), 1);
        assert_eq!(decision.tool_calls[0].server, "S");
    }

    #[test]
    fn parses_official_planned_tools_shape() {
        let response = r#"{"reasoning":"r","should_continue":true,"planned_tools":[{"tool":"Server A:tool_x","parameters":{"x":1}}]}"#;
        let decision = parse_planning_decision(response).unwrap();
        assert_eq!(decision.tool_calls[0].server, "Server A");
        assert_eq!(decision.tool_calls[0].tool, "tool_x");
    }

    #[test]
    fn normalizes_unique_tool_without_server() {
        let mut tools = BTreeMap::new();
        tools.insert(
            "Server A:tool_x".to_string(),
            McpBenchAvailableTool {
                name: "tool_x".to_string(),
                original_name: "tool_x".to_string(),
                server: "Server A".to_string(),
                description: String::new(),
                input_schema: json!({}),
            },
        );
        let call = PlannedToolCall {
            server: String::new(),
            tool: "tool_x".to_string(),
            arguments: Default::default(),
        };
        let normalized = normalize_planned_tool_call(&call, &tools).unwrap();
        assert_eq!(normalized.server, "Server A");
    }
}
