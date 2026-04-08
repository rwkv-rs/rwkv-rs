use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};
use sonic_rs::{Object as Map, Value};

use crate::cores::datasets::function_calling::{FunctionCall, ToolRequestor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauTask {
    pub id: String,
    pub description: Option<TaskDescription>,
    pub user_scenario: UserScenario,
    pub ticket: Option<String>,
    pub initial_state: Option<InitialState>,
    pub evaluation_criteria: Option<EvaluationCriteria>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDescription {
    pub purpose: Option<String>,
    pub relevant_policies: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserScenario {
    pub persona: Option<String>,
    pub instructions: UserInstructions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserInstructions {
    Structured(StructuredUserInstructions),
    Text(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredUserInstructions {
    pub domain: String,
    pub reason_for_call: String,
    pub known_info: Option<String>,
    pub unknown_info: Option<String>,
    pub task_instructions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialState {
    pub initialization_data: Option<InitializationData>,
    pub initialization_actions: Option<Vec<EnvFunctionCall>>,
    pub message_history: Option<Vec<RawMessage>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationData {
    pub agent_data: Option<Value>,
    pub user_data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvFunctionCall {
    pub env_type: ToolRequestor,
    pub func_name: String,
    #[serde(default)]
    pub arguments: Map,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<FunctionCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationCriteria {
    pub actions: Option<Vec<ExpectedAction>>,
    pub env_assertions: Option<Vec<EnvAssertion>>,
    pub communicate_info: Option<Vec<String>>,
    pub nl_assertions: Option<Vec<String>>,
    #[serde(default = "default_reward_basis")]
    pub reward_basis: Vec<RewardType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvAssertion {
    pub env_type: ToolRequestor,
    pub func_name: String,
    #[serde(default)]
    pub arguments: Map,
    #[serde(default = "default_assert_value")]
    pub assert_value: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
