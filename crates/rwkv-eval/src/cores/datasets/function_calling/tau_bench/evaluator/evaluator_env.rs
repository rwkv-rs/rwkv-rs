use std::{collections::BTreeMap, path::Path};

use super::super::{
    data_model::simulation::{DBCheck, EnvAssertionCheck, RewardInfo},
    domains::{TauDomain, canonical_db, create_domain_env, initialize_env},
    evaluator::evaluator_base::EvaluatorBase,
};
use crate::cores::datasets::function_calling::tau_bench::{RewardType, TauTask};

pub struct EnvironmentEvaluator;

impl EnvironmentEvaluator {
    pub fn calculate_reward(
        domain: TauDomain,
        dataset_root: &Path,
        task: &TauTask,
        full_trajectory: &[crate::cores::datasets::function_calling::tau_bench::Message],
    ) -> Result<RewardInfo, String> {
        if task.evaluation_criteria.is_none() {
            return Ok(RewardInfo::new(1.0).with_info_note("No evaluation criteria"));
        }

        let expected_actions = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.actions.as_ref());
        let env_assertions = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.env_assertions.as_ref());
        if expected_actions.is_none() && env_assertions.is_none() {
            return Ok(RewardInfo {
                reward: 1.0,
                db_check: Some(DBCheck {
                    db_match: true,
                    db_reward: 1.0,
                }),
                info: Some(sonic_rs::json!({ "note": "No expected actions or env assertions" })),
                ..RewardInfo::new(1.0)
            });
        }

        let predicted_tool_calls = EvaluatorBase::extract_predicted_tool_calls(full_trajectory);
        let mut predicted_environment = create_domain_env(domain, dataset_root)?;
        initialize_env(predicted_environment.as_mut(), task)?;
        for tool_call in &predicted_tool_calls {
            let _ = predicted_environment.execute_tool_call(tool_call);
        }

        let mut gold_environment = create_domain_env(domain, dataset_root)?;
        initialize_env(gold_environment.as_mut(), task)?;
        let golden_actions = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.actions.as_deref())
            .unwrap_or(&[]);
        for action in golden_actions {
            let _ = gold_environment.execute_tool_call(
                &crate::cores::datasets::function_calling::FunctionCall {
                    requestor: action.requestor,
                    name: action.name.clone(),
                    arguments: action.arguments.clone(),
                },
            );
        }

        let db_match = canonical_db(gold_environment.agent_db())
            == canonical_db(predicted_environment.agent_db())
            && canonical_db(gold_environment.user_db())
                == canonical_db(predicted_environment.user_db());
        let db_reward = if db_match { 1.0 } else { 0.0 };
        let db_check = DBCheck {
            db_match,
            db_reward,
        };

        let env_assertions = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.env_assertions.as_deref())
            .unwrap_or(&[]);
        let mut env_assertion_reward = 1.0;
        let mut env_assertion_checks = Vec::with_capacity(env_assertions.len());
        for env_assertion in env_assertions {
            let met = predicted_environment.run_env_assertion(env_assertion)?
                == env_assertion.assert_value;
            let reward = if met { 1.0 } else { 0.0 };
            env_assertion_reward *= reward;
            env_assertion_checks.push(EnvAssertionCheck {
                env_assertion: env_assertion.clone(),
                met,
                reward,
            });
        }

        let mut reward = 1.0;
        let mut reward_breakdown = BTreeMap::new();
        let reward_basis = &task.evaluation_criteria.as_ref().unwrap().reward_basis;
        if reward_basis.contains(&RewardType::Db) {
            reward *= db_reward;
            reward_breakdown.insert(RewardType::Db, db_reward);
        }
        if reward_basis.contains(&RewardType::EnvAssertion) {
            reward *= env_assertion_reward;
            reward_breakdown.insert(RewardType::EnvAssertion, env_assertion_reward);
        }

        Ok(RewardInfo {
            reward,
            db_check: Some(db_check),
            env_assertions: Some(env_assertion_checks),
            reward_basis: Some(reward_basis.clone()),
            reward_breakdown: Some(reward_breakdown),
            ..RewardInfo::new(reward)
        })
    }
}
