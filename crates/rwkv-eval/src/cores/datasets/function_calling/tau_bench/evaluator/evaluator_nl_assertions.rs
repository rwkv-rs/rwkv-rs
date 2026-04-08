use async_openai::{Client, config::OpenAIConfig};
use serde::{Deserialize, Serialize};
use sonic_rs::{Value, json};

use super::super::data_model::{
    message::Message,
    simulation::{NLAssertionCheck, RewardInfo},
};
use crate::cores::datasets::function_calling::tau_bench::{RewardType, TauTask};

#[derive(Debug, Serialize)]
struct NLAssertionsJudgeRequest<'a> {
    model: &'a str,
    messages: Vec<NLAssertionsJudgeMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    response_format: NLAssertionsJudgeResponseFormat,
}

#[derive(Debug, Serialize)]
struct NLAssertionsJudgeMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct NLAssertionsJudgeResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: NLAssertionsJudgeJsonSchema,
}

#[derive(Debug, Serialize)]
struct NLAssertionsJudgeJsonSchema {
    description: Option<String>,
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
struct NLAssertionsJudgeResponse {
    choices: Vec<NLAssertionsJudgeChoice>,
}

#[derive(Debug, Deserialize)]
struct NLAssertionsJudgeChoice {
    finish_reason: Option<String>,
    message: NLAssertionsJudgeResponseMessage,
}

#[derive(Debug, Deserialize)]
struct NLAssertionsJudgeResponseMessage {
    content: Option<String>,
    refusal: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct NLAssertionsJudgeResult {
    results: Vec<NLAssertionJudgeItem>,
}

#[derive(Debug, Deserialize)]
struct NLAssertionJudgeItem {
    #[serde(rename = "expectedOutcome")]
    expected_outcome: String,
    #[serde(rename = "metExpectation")]
    met_expectation: bool,
    reasoning: String,
}

pub struct NLAssertionsEvaluator;

impl NLAssertionsEvaluator {
    pub async fn calculate_reward(
        task: &TauTask,
        full_trajectory: &[Message],
        judger_client: &Client<OpenAIConfig>,
        judger_model_name: &str,
    ) -> Result<RewardInfo, String> {
        if task.evaluation_criteria.is_none() {
            return Ok(RewardInfo::new(1.0).with_info_note("No evaluation criteria"));
        }
        let nl_assertions = task
            .evaluation_criteria
            .as_ref()
            .and_then(|criteria| criteria.nl_assertions.as_ref());
        let Some(nl_assertions) = nl_assertions else {
            return Ok(RewardInfo {
                reward: 1.0,
                nl_assertions: Some(Vec::new()),
                reward_breakdown: Some(std::collections::BTreeMap::from([(RewardType::NlAssertion, 1.0)])),
                info: Some(json!({ "note": "No nl_assertions to evaluate" })),
                ..RewardInfo::new(1.0)
            });
        };

        let nl_assertions_checks = Self::evaluate_nl_assertions(
            full_trajectory,
            nl_assertions,
            judger_client,
            judger_model_name,
        )
        .await?;
        let reward = if nl_assertions_checks.iter().all(|result| result.met) {
            1.0
        } else {
            0.0
        };

        Ok(RewardInfo {
            reward,
            nl_assertions: Some(nl_assertions_checks),
            reward_breakdown: Some(std::collections::BTreeMap::from([(RewardType::NlAssertion, reward)])),
            ..RewardInfo::new(reward)
        })
    }

    async fn evaluate_nl_assertions(
        trajectory: &[Message],
        nl_assertions: &[String],
        judger_client: &Client<OpenAIConfig>,
        judger_model_name: &str,
    ) -> Result<Vec<NLAssertionCheck>, String> {
        let trajectory_str = trajectory
            .iter()
            .map(Message::to_nl_assertion_line)
            .collect::<Vec<_>>()
            .join("\n");

        let system_prompt = r#"
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with one field named `results`.
- `results` must be an array of objects with:
  - `expectedOutcome`: the expectation being graded
  - `reasoning`: a short explanation
  - `metExpectation`: true if satisfied, false otherwise
"#;
        let user_prompt = format!(
            "conversation:\n{trajectory_str}\n\nexpectedOutcomes:\n{}",
            sonic_rs::to_string_pretty(nl_assertions).map_err(|err| err.to_string())?
        );

        let req = NLAssertionsJudgeRequest {
            model: judger_model_name,
            messages: vec![
                NLAssertionsJudgeMessage {
                    role: "system",
                    content: system_prompt,
                },
                NLAssertionsJudgeMessage {
                    role: "user",
                    content: &user_prompt,
                },
            ],
            temperature: 0.0,
            top_p: 1.0,
            max_completion_tokens: 512,
            response_format: NLAssertionsJudgeResponseFormat {
                kind: "json_schema",
                json_schema: NLAssertionsJudgeJsonSchema {
                    description: None,
                    name: "tau_bench_nl_assertions".to_string(),
                    schema: json!({
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "expectedOutcome": { "type": "string" },
                                        "reasoning": { "type": "string" },
                                        "metExpectation": { "type": "boolean" }
                                    },
                                    "required": ["expectedOutcome", "reasoning", "metExpectation"],
                                    "additionalProperties": false
                                }
                            }
                        },
                        "required": ["results"],
                        "additionalProperties": false
                    }),
                    strict: true,
                },
            },
        };

        let response: NLAssertionsJudgeResponse = judger_client
            .chat()
            .create_byot(&req)
            .await
            .map_err(|err| format!("request failed: {err}"))?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| "judge returned no choices".to_string())?;
        let content = choice.message.content.clone().ok_or_else(|| {
            format!(
                "judge returned no content; refusal={:?}; finish_reason={:?}",
                choice.message.refusal, choice.finish_reason
            )
        })?;
        let result: NLAssertionsJudgeResult = sonic_rs::from_str(&content)
            .map_err(|err| format!("invalid judge json: {err}; content={content:?}"))?;
        if result.results.len() != nl_assertions.len() {
            return Err(format!(
                "judge returned {} results for {} nl assertions",
                result.results.len(),
                nl_assertions.len()
            ));
        }
        for (index, (expected_assertion, actual_result)) in nl_assertions
            .iter()
            .zip(result.results.iter())
            .enumerate()
        {
            if actual_result.expected_outcome != *expected_assertion {
                return Err(format!(
                    "judge result mismatch at index {index}: expected {:?}, got {:?}",
                    expected_assertion, actual_result.expected_outcome
                ));
            }
        }
        Ok(result
            .results
            .into_iter()
            .map(|result| NLAssertionCheck {
                nl_assertion: result.expected_outcome,
                met: result.met_expectation,
                justification: result.reasoning,
            })
            .collect())
    }
}
