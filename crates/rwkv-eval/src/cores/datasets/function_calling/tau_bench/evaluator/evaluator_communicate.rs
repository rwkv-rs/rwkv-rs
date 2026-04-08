use std::collections::BTreeMap;

use super::super::data_model::simulation::{CommunicateCheck, RewardInfo};
use crate::cores::datasets::function_calling::tau_bench::{Message, RewardType};

pub struct CommunicateEvaluator;

impl CommunicateEvaluator {
    pub fn calculate_reward(
        task: &crate::cores::datasets::function_calling::tau_bench::TauTask,
        full_trajectory: &[Message],
    ) -> RewardInfo {
        if task.evaluation_criteria.is_none() {
            return RewardInfo::new(1.0).with_info_note("No evaluation criteria");
        }
        let communicate_info = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.communicate_info.as_ref());
        let Some(communicate_info) = communicate_info else {
            return RewardInfo {
                reward: 1.0,
                reward_breakdown: Some(BTreeMap::from([(RewardType::Communicate, 1.0)])),
                info: Some(sonic_rs::json!({ "note": "No communicate_info to evaluate" })),
                ..RewardInfo::new(1.0)
            };
        };

        let communicate_checks = Self::evaluate_communicate_info(full_trajectory, communicate_info);
        let reward = if communicate_checks.iter().all(|result| result.met) {
            1.0
        } else {
            0.0
        };

        RewardInfo {
            reward,
            communicate_checks: Some(communicate_checks),
            reward_breakdown: Some(BTreeMap::from([(RewardType::Communicate, reward)])),
            ..RewardInfo::new(reward)
        }
    }

    fn evaluate_communicate_info(
        full_trajectory: &[Message],
        communicate_info: &[String],
    ) -> Vec<CommunicateCheck> {
        if communicate_info.is_empty() {
            return Vec::new();
        }

        let mut outputs = Vec::new();
        for info_str in communicate_info {
            let found_message = full_trajectory.iter().find_map(|message| {
                let message = message.assistant_message()?;
                if !message.has_text_content() {
                    return None;
                }
                let content = message.content.as_deref()?;
                let normalized_content = content.to_lowercase().replace(',', "");
                let normalized_info = info_str.to_lowercase().replace(',', "");
                normalized_content
                    .contains(&normalized_info)
                    .then_some(content.to_string())
            });
            let (met, justification) = match found_message {
                Some(message) => (
                    true,
                    format!("Information '{info_str}' communicated in the message:\n '{message}'"),
                ),
                None => (false, format!("Information '{info_str}' not communicated.")),
            };
            outputs.push(CommunicateCheck {
                info: info_str.clone(),
                met,
                justification,
            });
        }
        outputs
    }
}
