use async_openai::Client;
use async_openai::config::OpenAIConfig;
use serde::{Deserialize, Serialize};
use sonic_rs::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckerOutput {
    pub answer_correct: bool,
    pub instruction_following_error: bool,
    pub world_knowledge_error: bool,
    pub math_error: bool,
    pub reasoning_logic_error: bool,
    pub thought_contains_correct_answer: bool,
    pub reason: String,
}

impl CheckerOutput {
    pub fn needs_human_review(&self) -> bool {
        self.answer_correct || self.thought_contains_correct_answer
    }
}

#[derive(Debug, Serialize)]
struct CheckerRequest<'a> {
    model: &'a str,
    messages: Vec<CheckerMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    response_format: CheckerResponseFormat,
}

#[derive(Debug, Serialize)]
struct CheckerMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct CheckerResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: CheckerJsonSchema,
}

#[derive(Debug, Serialize)]
struct CheckerJsonSchema {
    description: Option<String>,
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
struct CheckerResponse {
    choices: Vec<CheckerChoice>,
}

#[derive(Debug, Deserialize)]
struct CheckerChoice {
    finish_reason: Option<String>,
    message: CheckerResponseMessage,
}

#[derive(Debug, Deserialize)]
struct CheckerResponseMessage {
    content: Option<String>,
    refusal: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CheckerWire {
    answer_correct: bool,
    instruction_following_error: bool,
    world_knowledge_error: bool,
    math_error: bool,
    reasoning_logic_error: bool,
    thought_contains_correct_answer: bool,
    reason: String,
}

pub async fn run_checker(
    checker_client: &Client<OpenAIConfig>,
    checker_model_name: &str,
    context: &str,
    answer: &str,
    ref_answer: &str,
) -> Result<CheckerOutput, String> {
    let user_prompt = format!(
        concat!(
            "[Rendered Context]\n{context}\n\n",
            "[Extracted Answer]\n{answer}\n\n",
            "[Reference Answer]\n{ref_answer}\n\n",
        ),
        context = context,
        answer = answer,
        ref_answer = ref_answer,
    );

    let req = CheckerRequest {
        model: checker_model_name,
        messages: vec![
            CheckerMessage {
                role: "system",
                content: concat!(
                    "You are a rigorous evaluation checker.\n",
                    "The main evaluator already marked the answer incorrect.\n",
                    "Review the rendered context, the extracted answer, and the reference answer.\n",
                    "Return only JSON matching the provided schema."
                ),
            },
            CheckerMessage {
                role: "user",
                content: &user_prompt,
            },
        ],
        temperature: 0.0,
        top_p: 1.0,
        max_completion_tokens: 256,
        response_format: CheckerResponseFormat {
            kind: "json_schema",
            json_schema: CheckerJsonSchema {
                description: None,
                name: "eval_checker".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "answer_correct": { "type": "boolean" },
                        "instruction_following_error": { "type": "boolean" },
                        "world_knowledge_error": { "type": "boolean" },
                        "math_error": { "type": "boolean" },
                        "reasoning_logic_error": { "type": "boolean" },
                        "thought_contains_correct_answer": { "type": "boolean" },
                        "reason": { "type": "string" }
                    },
                    "required": [
                        "answer_correct",
                        "instruction_following_error",
                        "world_knowledge_error",
                        "math_error",
                        "reasoning_logic_error",
                        "thought_contains_correct_answer",
                        "reason"
                    ],
                    "additionalProperties": false
                }),
                strict: true,
            },
        },
    };

    let resp: CheckerResponse = checker_client
        .chat()
        .create_byot(&req)
        .await
        .map_err(|err| format!("checker request failed: {err}"))?;
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| "checker returned no choices".to_string())?;
    let content = choice.message.content.clone().ok_or_else(|| {
        format!(
            "checker returned no content; refusal={:?}; finish_reason={:?}",
            choice.message.refusal, choice.finish_reason
        )
    })?;
    let wire = sonic_rs::from_str::<CheckerWire>(&content)
        .map_err(|err| format!("invalid checker json: {err}; content={content:?}"))?;
    let reason = wire.reason.trim().to_string();
    if reason.is_empty() {
        return Err("checker returned empty `reason`".to_string());
    }

    Ok(CheckerOutput {
        answer_correct: wire.answer_correct,
        instruction_following_error: wire.instruction_following_error,
        world_knowledge_error: wire.world_knowledge_error,
        math_error: wire.math_error,
        reasoning_logic_error: wire.reasoning_logic_error,
        thought_contains_correct_answer: wire.thought_contains_correct_answer,
        reason,
    })
}
