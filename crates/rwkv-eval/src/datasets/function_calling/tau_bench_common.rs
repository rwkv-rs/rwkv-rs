use super::{FunctionCall, ToolRequestor};
use serde::{Deserialize, Serialize};
use sonic_rs::{Array, Object as Map, Value, prelude::*};
use std::fmt::{self, Display};
use std::path::Path;

use super::tau_bench_airline::AirlineEnv;
use super::tau_bench_retail::RetailEnv;
use super::tau_bench_telecom::TelecomEnv;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TauDomain {
    Airline,
    Retail,
    Telecom,
}

impl TauDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Airline => "airline",
            Self::Retail => "retail",
            Self::Telecom => "telecom",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolArgSpec {
    pub name: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    pub requestor: ToolRequestor,
    pub name: &'static str,
    pub description: &'static str,
    pub arguments: &'static [ToolArgSpec],
}

#[derive(Debug, Clone, Deserialize)]
pub struct TauTask {
    pub id: String,
    pub description: Option<TaskDescription>,
    pub user_scenario: UserScenario,
    pub ticket: Option<String>,
    pub initial_state: Option<InitialState>,
    pub evaluation_criteria: Option<EvaluationCriteria>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TaskDescription {
    pub purpose: Option<String>,
    pub relevant_policies: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UserScenario {
    pub persona: Option<String>,
    pub instructions: UserInstructions,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum UserInstructions {
    Structured(StructuredUserInstructions),
    Text(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct StructuredUserInstructions {
    pub domain: String,
    pub reason_for_call: String,
    pub known_info: Option<String>,
    pub unknown_info: Option<String>,
    pub task_instructions: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InitialState {
    pub initialization_data: Option<InitializationData>,
    pub initialization_actions: Option<Vec<EnvFunctionCall>>,
    pub message_history: Option<Vec<RawMessage>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InitializationData {
    pub agent_data: Option<Value>,
    pub user_data: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnvFunctionCall {
    pub env_type: ToolRequestor,
    pub func_name: String,
    #[serde(default)]
    pub arguments: Map,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<FunctionCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationCriteria {
    pub actions: Option<Vec<ExpectedAction>>,
    pub env_assertions: Option<Vec<EnvAssertion>>,
    pub communicate_info: Option<Vec<String>>,
    pub nl_assertions: Option<Vec<String>>,
    #[serde(default = "default_reward_basis")]
    pub reward_basis: Vec<RewardType>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedAction {
    pub action_id: String,
    #[serde(default = "default_requestor")]
    pub requestor: ToolRequestor,
    pub name: String,
    #[serde(default)]
    pub arguments: Map,
    pub info: Option<String>,
    pub compare_args: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnvAssertion {
    pub env_type: ToolRequestor,
    pub func_name: String,
    #[serde(default)]
    pub arguments: Map,
    #[serde(default = "default_assert_value")]
    pub assert_value: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum RewardType {
    #[serde(rename = "DB")]
    Db,
    #[serde(rename = "ENV_ASSERTION")]
    EnvAssertion,
    #[serde(rename = "NL_ASSERTION")]
    NlAssertion,
    #[serde(rename = "ACTION")]
    Action,
    #[serde(rename = "COMMUNICATE")]
    Communicate,
}

fn default_reward_basis() -> Vec<RewardType> {
    vec![RewardType::Db, RewardType::Communicate]
}

fn default_requestor() -> ToolRequestor {
    ToolRequestor::Assistant
}

fn default_assert_value() -> bool {
    true
}

impl Display for UserInstructions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structured(value) => {
                writeln!(f, "Domain: {}", value.domain)?;
                writeln!(f, "Reason for call:\n\t{}", value.reason_for_call)?;
                if let Some(known) = &value.known_info {
                    writeln!(f, "Known info:\n\t{known}")?;
                }
                if let Some(unknown) = &value.unknown_info {
                    writeln!(f, "Unknown info:\n\t{unknown}")?;
                }
                write!(f, "Task instructions:\n\t{}", value.task_instructions)
            }
            Self::Text(value) => write!(f, "{value}"),
        }
    }
}

impl Display for UserScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(persona) = &self.persona {
            writeln!(f, "Persona:\n\t{persona}")?;
        }
        write!(f, "Instructions:\n\t{}", self.instructions)
    }
}

pub fn render_user_prompt(task: &TauTask) -> String {
    if let Some(ticket) = &task.ticket {
        return ticket.trim().to_string();
    }

    let mut text = task.user_scenario.to_string();
    if let Some(description) = &task.description {
        if let Some(purpose) = &description.purpose {
            text.push_str("\n\nPurpose:\n");
            text.push_str(purpose);
        }
        if let Some(notes) = &description.notes {
            text.push_str("\n\nNotes:\n");
            text.push_str(notes);
        }
    }
    text
}

pub fn render_system_prompt(
    policy: &str,
    assistant_tools: &[ToolSpec],
    user_tools: &[ToolSpec],
) -> String {
    let mut lines = vec![
        "You are solving a tau_bench function-calling task.".to_string(),
        "Follow the policy exactly.".to_string(),
        "When you need to execute a tool, respond with only one JSON object wrapped in a ```json fenced block.".to_string(),
        "The JSON shape is {\"requestor\":\"assistant|user\",\"name\":\"tool_name\",\"arguments\":{...}}.".to_string(),
        "Use requestor=user only when simulating a concrete customer-side action or inspection step.".to_string(),
        "When the task is complete, respond with a plain natural-language final answer and do not output JSON.".to_string(),
        String::new(),
        "Policy:".to_string(),
        policy.trim().to_string(),
        String::new(),
        "Available assistant tools:".to_string(),
    ];

    for tool in assistant_tools {
        lines.push(render_tool_spec(tool));
    }
    if !user_tools.is_empty() {
        lines.push(String::new());
        lines.push("Available user tools:".to_string());
        for tool in user_tools {
            lines.push(render_tool_spec(tool));
        }
    }

    lines.join("\n")
}

fn render_tool_spec(tool: &ToolSpec) -> String {
    let mut line = format!(
        "- {}.{}: {}",
        tool.requestor.as_str(),
        tool.name,
        tool.description
    );
    if !tool.arguments.is_empty() {
        line.push_str(" Args: ");
        line.push_str(
            &tool
                .arguments
                .iter()
                .map(|arg| format!("{}={}", arg.name, arg.description))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    line
}

pub trait TauDomainEnv: Send + Sync {
    fn policy(&self) -> &str;
    fn assistant_tools(&self) -> &'static [ToolSpec];
    fn user_tools(&self) -> &'static [ToolSpec];
    fn update_agent_data(&mut self, data: &Value) -> Result<(), String>;
    fn update_user_data(&mut self, data: &Value) -> Result<(), String>;
    fn run_env_function(&mut self, action: &EnvFunctionCall) -> Result<Value, String>;
    fn execute_tool_call(&mut self, tool_call: &FunctionCall) -> Result<Value, String>;
    fn run_env_assertion(&self, assertion: &EnvAssertion) -> Result<bool, String>;
    fn agent_db(&self) -> Option<Value>;
    fn user_db(&self) -> Option<Value>;
}

pub fn create_domain_env(
    domain: TauDomain,
    dataset_root: &Path,
) -> Result<Box<dyn TauDomainEnv>, String> {
    match domain {
        TauDomain::Airline => Ok(Box::new(AirlineEnv::load(dataset_root)?)),
        TauDomain::Retail => Ok(Box::new(RetailEnv::load(dataset_root)?)),
        TauDomain::Telecom => Ok(Box::new(TelecomEnv::load(dataset_root)?)),
    }
}

pub fn evaluate_task(
    domain: TauDomain,
    dataset_root: &Path,
    task: &TauTask,
    tool_calls: &[FunctionCall],
    assistant_messages: &[String],
) -> Result<bool, String> {
    let mut predicted = create_domain_env(domain, dataset_root)?;
    initialize_env(predicted.as_mut(), task)?;
    for tool_call in tool_calls {
        let _ = predicted.execute_tool_call(tool_call);
    }

    let mut reward = 1.0_f32;
    let criteria = match &task.evaluation_criteria {
        Some(value) => value,
        None => return Ok(true),
    };

    if criteria.reward_basis.contains(&RewardType::Db) {
        let mut gold = create_domain_env(domain, dataset_root)?;
        initialize_env(gold.as_mut(), task)?;
        for action in criteria.actions.as_deref().unwrap_or(&[]) {
            let _ = gold.execute_tool_call(&FunctionCall {
                requestor: action.requestor,
                name: action.name.clone(),
                arguments: action.arguments.clone(),
            });
        }
        let db_match = canonical_db(gold.agent_db()) == canonical_db(predicted.agent_db())
            && canonical_db(gold.user_db()) == canonical_db(predicted.user_db());
        reward *= if db_match { 1.0 } else { 0.0 };
    }

    if criteria.reward_basis.contains(&RewardType::EnvAssertion) {
        for assertion in criteria.env_assertions.as_deref().unwrap_or(&[]) {
            let ok = predicted.run_env_assertion(assertion)? == assertion.assert_value;
            reward *= if ok { 1.0 } else { 0.0 };
        }
    }

    if criteria.reward_basis.contains(&RewardType::Action) {
        let all_actions_met = criteria
            .actions
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .all(|expected| expected.matches(tool_calls));
        reward *= if all_actions_met { 1.0 } else { 0.0 };
    }

    if criteria.reward_basis.contains(&RewardType::Communicate) {
        let communicate_ok = criteria
            .communicate_info
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .all(|needle| {
                let needle = needle.to_lowercase().replace(',', "");
                assistant_messages
                    .iter()
                    .any(|message| message.to_lowercase().replace(',', "").contains(&needle))
            });
        reward *= if communicate_ok { 1.0 } else { 0.0 };
    }

    Ok(reward > 0.0)
}

fn initialize_env(env: &mut dyn TauDomainEnv, task: &TauTask) -> Result<(), String> {
    if let Some(initial_state) = &task.initial_state {
        if let Some(data) = &initial_state.initialization_data {
            if let Some(agent_data) = &data.agent_data {
                env.update_agent_data(agent_data)?;
            }
            if let Some(user_data) = &data.user_data {
                env.update_user_data(user_data)?;
            }
        }
        for action in initial_state
            .initialization_actions
            .as_deref()
            .unwrap_or(&[])
        {
            let _ = env.run_env_function(action)?;
        }
    }
    Ok(())
}

impl ExpectedAction {
    pub fn matches(&self, tool_calls: &[FunctionCall]) -> bool {
        tool_calls.iter().any(|tool_call| {
            if self.requestor != tool_call.requestor || self.name != tool_call.name {
                return false;
            }
            match &self.compare_args {
                Some(keys) if !keys.is_empty() => keys
                    .iter()
                    .all(|key| self.arguments.get(&key) == tool_call.arguments.get(&key)),
                _ => self.arguments == tool_call.arguments,
            }
        })
    }
}

pub fn canonical_db(value: Option<Value>) -> Option<String> {
    value.map(|value| sonic_rs::to_string(&value).unwrap())
}

pub fn as_object(value: &Value) -> Result<&Map, String> {
    value
        .as_object()
        .ok_or_else(|| "expected json object".to_string())
}

pub fn as_object_mut(value: &mut Value) -> Result<&mut Map, String> {
    value
        .as_object_mut()
        .ok_or_else(|| "expected json object".to_string())
}

pub fn as_array(value: &Value) -> Result<&Array, String> {
    value
        .as_array()
        .ok_or_else(|| "expected json array".to_string())
}

pub fn as_array_mut(value: &mut Value) -> Result<&mut Array, String> {
    value
        .as_array_mut()
        .ok_or_else(|| "expected json array".to_string())
}

pub fn get_string_field<'a>(object: &'a Map, key: &str) -> Result<&'a str, String> {
    object
        .get(&key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field `{key}`"))
}

pub fn get_bool_field(object: &Map, key: &str) -> Result<bool, String> {
    object
        .get(&key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing bool field `{key}`"))
}

pub fn get_f64_field(object: &Map, key: &str) -> Result<f64, String> {
    object
        .get(&key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing number field `{key}`"))
}

pub fn get_i64_field(object: &Map, key: &str) -> Result<i64, String> {
    object
        .get(&key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing integer field `{key}`"))
}

pub fn get_value<'a>(object: &'a Map, key: &str) -> Result<&'a Value, String> {
    object
        .get(&key)
        .ok_or_else(|| format!("missing field `{key}`"))
}

pub fn update_json(target: &mut Value, patch: &Value) -> Result<(), String> {
    if let (Some(target), Some(patch)) = (target.as_object_mut(), patch.as_object()) {
        for (key, value) in patch {
            match target.get_mut(&key) {
                Some(existing) => update_json(existing, value)?,
                None => {
                    target.insert(&key, value.clone());
                }
            }
        }
        return Ok(());
    }

    *target = patch.clone();
    Ok(())
}

pub fn calculate_expression(expression: &str) -> Result<String, String> {
    if !expression
        .chars()
        .all(|char| "0123456789+-*/(). ".contains(char))
    {
        return Err("Invalid characters in expression".to_string());
    }
    let value = eval_expression(expression)?;
    let rendered = format!("{value:.2}");
    Ok(rendered
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string())
}

fn eval_expression(expression: &str) -> Result<f64, String> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Op {
        Add,
        Sub,
        Mul,
        Div,
        LParen,
    }

    impl Op {
        fn precedence(self) -> u8 {
            match self {
                Self::Add | Self::Sub => 1,
                Self::Mul | Self::Div => 2,
                Self::LParen => 0,
            }
        }
    }

    fn apply(values: &mut Vec<f64>, op: Op) -> Result<(), String> {
        if op == Op::LParen {
            return Ok(());
        }
        let rhs = values.pop().ok_or_else(|| "missing rhs".to_string())?;
        let lhs = values.pop().ok_or_else(|| "missing lhs".to_string())?;
        let value = match op {
            Op::Add => lhs + rhs,
            Op::Sub => lhs - rhs,
            Op::Mul => lhs * rhs,
            Op::Div => lhs / rhs,
            Op::LParen => unreachable!(),
        };
        values.push(value);
        Ok(())
    }

    let mut ops = Vec::<Op>::new();
    let mut values = Vec::<f64>::new();
    let chars = expression.as_bytes();
    let mut idx = 0;
    let mut expect_number = true;

    while idx < chars.len() {
        let ch = chars[idx] as char;
        if ch.is_ascii_whitespace() {
            idx += 1;
            continue;
        }

        if ch == '(' {
            ops.push(Op::LParen);
            idx += 1;
            expect_number = true;
            continue;
        }
        if ch == ')' {
            while let Some(op) = ops.pop() {
                if op == Op::LParen {
                    break;
                }
                apply(&mut values, op)?;
            }
            idx += 1;
            expect_number = false;
            continue;
        }

        if ch.is_ascii_digit() || ch == '.' || (expect_number && matches!(ch, '+' | '-')) {
            let start = idx;
            idx += 1;
            while idx < chars.len() {
                let next = chars[idx] as char;
                if next.is_ascii_digit() || next == '.' {
                    idx += 1;
                } else {
                    break;
                }
            }
            let number = expression[start..idx]
                .parse::<f64>()
                .map_err(|err| format!("invalid number `{}`: {err}", &expression[start..idx]))?;
            values.push(number);
            expect_number = false;
            continue;
        }

        let op = match ch {
            '+' => Op::Add,
            '-' => Op::Sub,
            '*' => Op::Mul,
            '/' => Op::Div,
            _ => return Err(format!("unsupported token `{ch}`")),
        };
        while let Some(last) = ops.last().copied() {
            if last.precedence() < op.precedence() || last == Op::LParen {
                break;
            }
            ops.pop();
            apply(&mut values, last)?;
        }
        ops.push(op);
        idx += 1;
        expect_number = true;
    }

    while let Some(op) = ops.pop() {
        apply(&mut values, op)?;
    }
    if values.len() != 1 {
        return Err("expression did not collapse to a single value".to_string());
    }
    Ok(values[0])
}
