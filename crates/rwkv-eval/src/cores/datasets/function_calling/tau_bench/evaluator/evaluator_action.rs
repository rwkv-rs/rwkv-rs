use std::collections::BTreeMap;

use super::super::{
    data_model::simulation::{ActionCheck, RewardInfo},
    evaluator::evaluator_base::EvaluatorBase,
};
use crate::cores::datasets::function_calling::tau_bench::RewardType;

pub struct ActionEvaluator;

impl ActionEvaluator {
    pub fn calculate_reward(
        task: &crate::cores::datasets::function_calling::tau_bench::TauTask,
        full_trajectory: &[crate::cores::datasets::function_calling::tau_bench::Message],
    ) -> RewardInfo {
        if task.evaluation_criteria.is_none() {
            return RewardInfo::new(1.0).with_info_note("No evaluation criteria");
        }
        let golden_actions = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.actions.as_ref());
        let Some(golden_actions) = golden_actions else {
            return RewardInfo {
                reward: 1.0,
                reward_breakdown: Some(BTreeMap::from([(RewardType::Action, 1.0)])),
                info: Some(sonic_rs::json!({ "note": "No actions to evaluate" })),
                ..RewardInfo::new(1.0)
            };
        };

        let predicted_tool_calls = EvaluatorBase::extract_predicted_tool_calls(full_trajectory);
        let action_checks = golden_actions
            .iter()
            .map(|gold_action| {
                let action_match = gold_action.matches(&predicted_tool_calls);
                ActionCheck {
                    action: gold_action.clone(),
                    action_match,
                    action_reward: if action_match { 1.0 } else { 0.0 },
                }
            })
            .collect::<Vec<_>>();
        let reward = if action_checks.iter().all(|result| result.action_match) {
            1.0
        } else {
            0.0
        };

        RewardInfo {
            reward,
            action_checks: Some(action_checks),
            reward_breakdown: Some(BTreeMap::from([(RewardType::Action, reward)])),
            ..RewardInfo::new(reward)
        }
    }
}
