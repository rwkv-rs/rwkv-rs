use std::path::Path;

use async_openai::{Client, config::OpenAIConfig};

use super::{
    evaluator_action::ActionEvaluator,
    evaluator_communicate::CommunicateEvaluator,
    evaluator_env::EnvironmentEvaluator,
    evaluator_nl_assertions::NLAssertionsEvaluator,
};
use crate::cores::datasets::function_calling::tau_bench::{
    EvaluationType,
    Message,
    RewardInfo,
    RewardType,
    TauDomain,
    TauTask,
};

pub struct EvaluationContext<'a> {
    pub domain: TauDomain,
    pub dataset_root: &'a Path,
    pub task: &'a TauTask,
    pub full_trajectory: &'a [Message],
    pub evaluation_type: EvaluationType,
    pub judger_model_name: Option<&'a str>,
    pub judger_client: Option<&'a Client<OpenAIConfig>>,
}

pub async fn evaluate_simulation(context: EvaluationContext<'_>) -> Result<RewardInfo, String> {
    if context.task.evaluation_criteria.is_none() {
        return Ok(RewardInfo::new(1.0).with_info_note("No evaluation criteria"));
    }

    match context.evaluation_type {
        EvaluationType::Env => EnvironmentEvaluator::calculate_reward(
            context.domain,
            context.dataset_root,
            context.task,
            context.full_trajectory,
        ),
        EvaluationType::Action => Ok(ActionEvaluator::calculate_reward(
            context.task,
            context.full_trajectory,
        )),
        EvaluationType::Communicate => Ok(CommunicateEvaluator::calculate_reward(
            context.task,
            context.full_trajectory,
        )),
        EvaluationType::NlAssertions => evaluate_nl_assertions(&context).await,
        EvaluationType::All | EvaluationType::AllWithNlAssertions => {
            let env_reward_info = EnvironmentEvaluator::calculate_reward(
                context.domain,
                context.dataset_root,
                context.task,
                context.full_trajectory,
            )?;
            let action_reward_info =
                ActionEvaluator::calculate_reward(context.task, context.full_trajectory);
            let communicate_reward_info =
                CommunicateEvaluator::calculate_reward(context.task, context.full_trajectory);
            let nl_reward_info = if context.evaluation_type == EvaluationType::AllWithNlAssertions {
                Some(evaluate_nl_assertions(&context).await?)
            } else {
                None
            };

            let mut reward = 1.0;
            let mut reward_breakdown = std::collections::BTreeMap::new();
            let task_reward_basis = &context
                .task
                .evaluation_criteria
                .as_ref()
                .unwrap()
                .reward_basis;

            if includes_env_basis(task_reward_basis) {
                if let Some(breakdown) = &env_reward_info.reward_breakdown {
                    reward_breakdown.extend(breakdown.clone());
                }
                reward *= env_reward_info.reward;
            }
            if task_reward_basis.contains(&RewardType::Action) {
                if let Some(breakdown) = &action_reward_info.reward_breakdown {
                    reward_breakdown.extend(breakdown.clone());
                }
                reward *= action_reward_info.reward;
            }
            if task_reward_basis.contains(&RewardType::NlAssertion) {
                if context.evaluation_type != EvaluationType::AllWithNlAssertions {
                    return Err(
                        "NL assertions are part of the reward basis, but they are not being evaluated."
                            .to_string(),
                    );
                }
                if let Some(nl_reward_info) = &nl_reward_info {
                    if let Some(breakdown) = &nl_reward_info.reward_breakdown {
                        reward_breakdown.extend(breakdown.clone());
                    }
                    reward *= nl_reward_info.reward;
                }
            }
            if task_reward_basis.contains(&RewardType::Communicate) {
                if let Some(breakdown) = &communicate_reward_info.reward_breakdown {
                    reward_breakdown.extend(breakdown.clone());
                }
                reward *= communicate_reward_info.reward;
            }

            let nl_assertions = nl_reward_info
                .as_ref()
                .and_then(|reward_info| reward_info.nl_assertions.clone());
            let nl_info = nl_reward_info
                .as_ref()
                .and_then(|reward_info| reward_info.info.clone());

            Ok(RewardInfo {
                reward,
                db_check: env_reward_info.db_check,
                env_assertions: env_reward_info.env_assertions,
                action_checks: action_reward_info.action_checks,
                nl_assertions,
                communicate_checks: communicate_reward_info.communicate_checks,
                reward_basis: Some(task_reward_basis.clone()),
                reward_breakdown: Some(reward_breakdown),
                info: Some(sonic_rs::json!({
                    "env": env_reward_info.info,
                    "nl": nl_info,
                    "communicate": communicate_reward_info.info,
                    "action": action_reward_info.info,
                })),
            })
        }
    }
}

fn includes_env_basis(task_reward_basis: &[RewardType]) -> bool {
    task_reward_basis.contains(&RewardType::Db)
        || task_reward_basis.contains(&RewardType::EnvAssertion)
}

async fn evaluate_nl_assertions(context: &EvaluationContext<'_>) -> Result<RewardInfo, String> {
    let judger_client = context
        .judger_client
        .ok_or_else(|| "nl assertion judge unavailable".to_string())?;
    let judger_model_name = context
        .judger_model_name
        .ok_or_else(|| "nl assertion judge unavailable".to_string())?;
    NLAssertionsEvaluator::calculate_reward(
        context.task,
        context.full_trajectory,
        judger_client,
        judger_model_name,
    )
    .await
}
