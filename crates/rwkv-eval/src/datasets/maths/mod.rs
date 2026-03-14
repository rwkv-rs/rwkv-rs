use crate::datasets::{
    CoTMode, SamplingConfig, apply_user_assistant_template, get_completions_of_cot,
};
use crate::inferers::{CompletionRequest, CompletionResponse};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage, CreateChatCompletionRequest, ResponseFormat,
    ResponseFormatJsonSchema,
};
use serde::Deserialize;
use serde_json::json;

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
pub mod simpleqa;
pub mod svamp;

pub fn get_expect_context(
    subject: &str,
    question: &str,
    cot_mode: CoTMode,
) -> String {
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
) -> String {
    if cot_mode != CoTMode::CoT {
        panic!("maths only supports CoT mode, got {cot_mode:?}");
    }

    let prompt_for_cot = get_prompt_for_cot(expected_context);
    let completions_of_cot =
        get_completions_of_cot(model_client, model_name, &prompt_for_cot, sampling_config).await;
    let prompt_for_final_answer =
        get_prompt_for_final_answer(expected_context, Some(&completions_of_cot));

    get_final_answer(
        model_client,
        model_name,
        &prompt_for_final_answer,
        sampling_config,
    )
    .await
}

async fn get_final_answer(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt_for_final_answer: &str,
    sampling_config: &SamplingConfig,
) -> String {
    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt_for_final_answer.into(),
        vec![],
        128,
        sampling_config,
        None,
        None,
    );

    let resp: CompletionResponse = model_client.completions().create_byot(&req).await.unwrap();
    resp.choices[0].text.clone()
}

pub async fn judge_with_retry(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    expected_context: &str,
    reference_answer: &str,
    model_final_answer: &str,
) -> bool {
    for attempt in 1..=3 {
        match judge_once(
            judger_client,
            judger_model_name,
            expected_context,
            reference_answer,
            model_final_answer,
        )
        .await
        {
            Ok(result) => return result,
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

    let req = CreateChatCompletionRequest {
        model: judger_model_name.to_string(),
        messages: vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: concat!(
                    "You are a rigorous math judge.\n",
                    "Decide whether the model final answer is correct with respect to the reference answer.\n",
                    "Use the expected context as the task definition and grading boundary.\n",
                    "Return only JSON matching the provided schema."
                )
                .into(),
                ..Default::default()
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: user_prompt.into(),
                ..Default::default()
            }),
        ],
        temperature: Some(0.0),
        top_p: Some(1.0),
        max_completion_tokens: Some(32),
        response_format: Some(ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                description: None,
                name: "maths_judge".to_string(),
                schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "is_passed": { "type": "boolean" }
                    },
                    "required": ["is_passed"],
                    "additionalProperties": false
                })),
                strict: Some(true),
            },
        }),
        ..Default::default()
    };

    let resp = judger_client
        .chat()
        .create(req)
        .await
        .map_err(|err| format!("request failed: {err}"))?;
    let message = resp
        .choices
        .first()
        .ok_or_else(|| "judge returned no choices".to_string())?
        .message
        .clone();
    let content = message.content.ok_or_else(|| {
        format!(
            "judge returned no content; refusal={:?}; finish_reason={:?}",
            message.refusal,
            resp.choices.first().and_then(|choice| choice.finish_reason)
        )
    })?;

    serde_json::from_str::<JudgeResult>(&content)
        .map(|result| result.is_passed)
        .map_err(|err| format!("invalid judge json: {err}; content={content:?}"))
}

#[derive(Deserialize)]
struct JudgeResult {
    is_passed: bool,
}
