use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sonic_rs::{JsonValueTrait, Value, json};

use super::tasks::{EnvAssertion, ExpectedAction, RewardType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationType {
    Env,
    Communicate,
    Action,
    All,
    NlAssertions,
    AllWithNlAssertions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLAssertionCheck {
    pub nl_assertion: String,
    pub met: bool,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicateCheck {
    pub info: String,
    pub met: bool,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBCheck {
    pub db_match: bool,
    pub db_reward: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCheck {
    pub action: ExpectedAction,
    pub action_match: bool,
    pub action_reward: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvAssertionCheck {
    pub env_assertion: EnvAssertion,
    pub met: bool,
    pub reward: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardInfo {
    pub reward: f32,
    pub db_check: Option<DBCheck>,
    pub env_assertions: Option<Vec<EnvAssertionCheck>>,
    pub action_checks: Option<Vec<ActionCheck>>,
    pub nl_assertions: Option<Vec<NLAssertionCheck>>,
    pub communicate_checks: Option<Vec<CommunicateCheck>>,
    pub reward_basis: Option<Vec<RewardType>>,
    pub reward_breakdown: Option<BTreeMap<RewardType, f32>>,
    pub info: Option<Value>,
}

impl RewardInfo {
    pub fn new(reward: f32) -> Self {
        Self {
            reward,
            db_check: None,
            env_assertions: None,
            action_checks: None,
            nl_assertions: None,
            communicate_checks: None,
            reward_basis: Some(vec![RewardType::Db]),
            reward_breakdown: None,
            info: None,
        }
    }

    pub fn failure_reason(&self) -> String {
        if let Some(db_check) = &self.db_check {
            if !db_check.db_match {
                return "db mismatch".to_string();
            }
        }
        if self
            .env_assertions
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .any(|assertion| !assertion.met)
        {
            return "env assertion failed".to_string();
        }
        if self
            .action_checks
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .any(|action| !action.action_match)
        {
            return "action expectation failed".to_string();
        }
        if self
            .nl_assertions
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .any(|assertion| !assertion.met)
        {
            return "nl assertion failed".to_string();
        }
        if self
            .communicate_checks
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .any(|check| !check.met)
        {
            return "communicate expectation failed".to_string();
        }
        if let Some(info) = &self.info {
            if let Some(note) = info.get("note").and_then(Value::as_str) {
                return note.to_string();
            }
        }
        "task evaluation returned false".to_string()
    }

    pub fn with_info_note(mut self, note: &str) -> Self {
        self.info = Some(json!({ "note": note }));
        self
    }
}
