mod airline;
mod retail;
mod telecom;

use std::{
    fmt::{self, Display},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sonic_rs::{
    Array,
    JsonContainerTrait,
    JsonValueMutTrait,
    JsonValueTrait,
    Object as Map,
    Value,
};

pub use super::data_model::tasks::{EnvAssertion, EnvFunctionCall, TauTask};
pub use crate::cores::datasets::function_calling::{FunctionCall, ToolRequestor};
use self::{airline::AirlineEnv, retail::RetailEnv, telecom::TelecomEnv};

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

pub fn initialize_env(env: &mut dyn TauDomainEnv, task: &TauTask) -> Result<(), String> {
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

impl Display for TauDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
