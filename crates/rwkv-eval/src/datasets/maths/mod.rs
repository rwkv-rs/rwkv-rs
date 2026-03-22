use crate::datasets::{
    CoTMode, SamplingConfig, apply_user_assistant_template, get_completions_of_cot, render_context,
};
use crate::inferers::generate_text_completion;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use sonic_rs::{Value, json, prelude::*};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub mod aime24;
pub mod aime25;
pub mod algebra222;
pub mod amc23;
pub mod answer_judge;
pub mod asdiv;
pub mod beyond_aime;
pub mod brumo25;
pub mod college_math;
pub mod comp_math_24_25;
pub mod gaokao2023en;
pub mod gsm8k;
pub mod gsm_plus;
pub mod hendrycks_math;
pub mod hle;
pub mod hmmt_feb25;
pub mod math_500;
pub mod math_odyssey;
pub mod mawps;
pub mod minerva_math;
pub mod olympiadbench;
pub mod omni_math;
pub mod polymath;
pub mod simpleqa;
pub mod svamp;

#[derive(Debug, Serialize)]
struct MathsJudgeRequest<'a> {
    model: &'a str,
    messages: Vec<MathsJudgeMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    response_format: MathsJudgeResponseFormat,
}

#[derive(Debug, Serialize)]
struct MathsJudgeMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct MathsJudgeResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: MathsJudgeJsonSchema,
}

#[derive(Debug, Serialize)]
struct MathsJudgeJsonSchema {
    description: Option<String>,
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
struct MathsJudgeResponse {
    choices: Vec<MathsJudgeChoice>,
}

#[derive(Debug, Deserialize)]
struct MathsJudgeChoice {
    finish_reason: Option<String>,
    message: MathsJudgeResponseMessage,
}

#[derive(Debug, Deserialize)]
struct MathsJudgeResponseMessage {
    content: Option<String>,
    refusal: Option<Value>,
}

pub struct FreeAnswerRecord {
    pub context: String,
    pub answer: String,
}

pub struct JudgeOutcome {
    pub is_passed: bool,
    pub fail_reason: String,
}

static LLM_JUDGER_SEMAPHORE: OnceCell<Arc<Semaphore>> = OnceCell::new();

pub fn set_llm_judger_semaphore(semaphore: Arc<Semaphore>) {
    let _ = LLM_JUDGER_SEMAPHORE.set(semaphore);
}

pub fn json_value_as_text(value: &Value) -> Option<String> {
    if value.is_null() {
        None
    } else if let Some(value) = value.as_str() {
        Some(value.trim().to_string())
    } else if let Some(value) = value.as_number() {
        Some(value.to_string())
    } else if let Some(value) = value.as_bool() {
        Some(value.to_string())
    } else {
        sonic_rs::to_string(value).ok()
    }
}

pub fn extract_last_boxed_answer(solution: &str) -> Option<String> {
    let marker = r"\boxed{";
    let mut search_end = solution.len();

    while let Some(start) = solution[..search_end].rfind(marker) {
        let content_start = start + marker.len();
        let tail = &solution[content_start..];
        let mut depth = 0usize;

        for (idx, ch) in tail.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    if depth == 0 {
                        return Some(tail[..idx].trim().to_string());
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }

        search_end = start;
    }

    None
}

pub fn get_expect_context(subject: &str, question: &str, cot_mode: CoTMode) -> String {
    if cot_mode != CoTMode::CoT {
        panic!("maths only supports CoT mode, got {cot_mode:?}");
    }

    let user_part = format!(
        concat!(
            "You are a very talented expert in {subject}.\n",
            "Solve the problem and output the final answer in \\boxed{{}}.\n",
            "Problem: {question}",
        ),
        subject = subject,
        question = question,
    );
    let assistant_part = concat!(
        "<think><|completions_of_cot|></think>\n",
        "Therefore, the answer is \\(\\boxed{<|final_answer|>}\\)."
    )
    .to_string();

    apply_user_assistant_template(user_part, assistant_part)
}

fn get_prompt_for_cot(expected_context: &str) -> String {
    expected_context
        .split_once("<|completions_of_cot|>")
        .unwrap()
        .0
        .to_string()
}

fn get_prompt_for_final_answer(expected_context: &str, completions_of_cot: Option<&str>) -> String {
    completions_of_cot
        .map(|cot| expected_context.replace("<|completions_of_cot|>", cot))
        .unwrap_or_else(|| expected_context.to_string())
        .split_once("<|final_answer|>")
        .unwrap_or_else(|| {
            panic!(
                "expected_context missing <|final_answer|> marker: {}",
                expected_context
            )
        })
        .0
        .to_string()
}

pub async fn get_final_answer_with_cot_mode(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    expected_context: &str,
    sampling_config: &SamplingConfig,
    cot_mode: CoTMode,
) -> FreeAnswerRecord {
    if cot_mode != CoTMode::CoT {
        panic!("maths only supports CoT mode, got {cot_mode:?}");
    }

    let prompt_for_cot = get_prompt_for_cot(expected_context);
    let completions_of_cot =
        get_completions_of_cot(model_client, model_name, &prompt_for_cot, sampling_config).await;
    let prompt_for_final_answer =
        get_prompt_for_final_answer(expected_context, Some(&completions_of_cot));
    let answer = get_final_answer(
        model_client,
        model_name,
        &prompt_for_final_answer,
        sampling_config,
    )
    .await;

    FreeAnswerRecord {
        context: render_context(
            expected_context,
            &[
                ("<|completions_of_cot|>", &completions_of_cot),
                ("<|final_answer|>", &answer),
            ],
        ),
        answer,
    }
}

async fn get_final_answer(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt_for_final_answer: &str,
    sampling_config: &SamplingConfig,
) -> String {
    generate_text_completion(
        model_client,
        model_name,
        &prompt_for_final_answer,
        vec![],
        128,
        sampling_config,
    )
    .await
    .unwrap()
}

pub async fn judge_with_retry(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    expected_context: &str,
    reference_answer: &str,
    model_final_answer: &str,
) -> JudgeOutcome {
    for attempt in 1..=3 {
        let _permit = match LLM_JUDGER_SEMAPHORE.get() {
            Some(semaphore) => Some(
                Arc::clone(semaphore)
                    .acquire_owned()
                    .await
                    .map_err(|err| format!("acquire judger semaphore failed: {err}"))
                    .unwrap_or_else(|err| panic!("{err}")),
            ),
            None => None,
        };
        match judge_once(
            judger_client,
            judger_model_name,
            expected_context,
            reference_answer,
            model_final_answer,
        )
        .await
        {
            Ok(is_passed) => {
                return JudgeOutcome {
                    is_passed,
                    fail_reason: if is_passed {
                        String::new()
                    } else {
                        "judger marked answer incorrect".to_string()
                    },
                };
            }
            Err(err) if attempt < 3 => {
                eprintln!(
                    "llm judge failed on attempt {attempt}/3, retrying: {err}; model={judger_model_name}"
                );
            }
            Err(err) => {
                panic!(
                    "llm judge failed after 3 attempts: {err}; model={judger_model_name}; reference={reference_answer}; answer={model_final_answer}"
                );
            }
        }
    }

    panic!("unreachable judge retry loop")
}

async fn judge_once(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    expected_context: &str,
    reference_answer: &str,
    model_final_answer: &str,
) -> Result<bool, String> {
    let user_prompt = format!(
        concat!(
            "[Expected Context]\n{expected_context}\n\n",
            "[Reference Answer]\n{reference_answer}\n\n",
            "[Model Final Answer]\n{model_final_answer}\n\n",
        ),
        expected_context = expected_context,
        reference_answer = reference_answer,
        model_final_answer = model_final_answer,
    );

    let req = MathsJudgeRequest {
        model: judger_model_name,
        messages: vec![
            MathsJudgeMessage {
                role: "system",
                content: concat!(
                    "You are a rigorous math judge.\n",
                    "Decide whether the model final answer is correct with respect to the reference answer.\n",
                    "Use the expected context as the task definition and grading boundary.\n",
                    "Return only JSON matching the provided schema."
                ),
            },
            MathsJudgeMessage {
                role: "user",
                content: &user_prompt,
            },
        ],
        temperature: 0.001,
        top_p: 1.0,
        max_completion_tokens: 32,
        response_format: MathsJudgeResponseFormat {
            kind: "json_schema",
            json_schema: MathsJudgeJsonSchema {
                description: None,
                name: "maths_judge".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "is_passed": { "type": "boolean" }
                    },
                    "required": ["is_passed"],
                    "additionalProperties": false
                }),
                strict: true,
            },
        },
    };

    let resp: MathsJudgeResponse = judger_client
        .chat()
        .create_byot(&req)
        .await
        .map_err(|err| format!("request failed: {err}"))?;
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| "judge returned no choices".to_string())?;
    let content = choice.message.content.clone().ok_or_else(|| {
        format!(
            "judge returned no content; refusal={:?}; finish_reason={:?}",
            choice.message.refusal, choice.finish_reason
        )
    })?;

    sonic_rs::from_str::<JudgeResult>(&content)
        .map(|result| result.is_passed)
        .map_err(|err| format!("invalid judge json: {err}; content={content:?}"))
}

#[derive(Deserialize)]
struct JudgeResult {
    is_passed: bool,
}

#[cfg(test)]
mod tests {
    use super::extract_last_boxed_answer;

    #[test]
    fn extracts_nested_boxed_answer() {
        let text = r"Therefore, the answer is \boxed{\frac{13}{4}}.";
        assert_eq!(
            extract_last_boxed_answer(text).as_deref(),
            Some(r"\frac{13}{4}")
        );
    }

    #[test]
    fn extracts_last_boxed_answer_when_multiple_exist() {
        let text = r"Scratch \boxed{wrong} and final \boxed{\sqrt{2}}";
        assert_eq!(
            extract_last_boxed_answer(text).as_deref(),
            Some(r"\sqrt{2}")
        );
    }
}
