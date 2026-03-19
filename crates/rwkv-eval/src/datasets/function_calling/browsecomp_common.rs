use crate::datasets::SamplingConfig;
use crate::datasets::function_calling::{
    build_turn_completion_prompt, get_completion, get_expected_context,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sonic_rs::{Value, json};

const BROWSECOMP_JUDGE_SAMPLING_CONFIG: SamplingConfig = SamplingConfig {
    temperature: 0.0,
    top_k: 1,
    top_p: 1.0,
    presence_penalty: 0.0,
    repetition_penalty: 0.0,
    penalty_decay: 1.0,
};

const BROWSECOMP_CONTROL_MARKERS: &[&str] = &[
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|completions|>",
    "<|completions_of_cot|>",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrowseCompLocale {
    En,
    Zh,
}

#[derive(Debug)]
pub struct BrowseJudgeOutcome {
    pub is_passed: bool,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
struct BrowseJudgeWire {
    is_passed: bool,
    reason: String,
}

#[derive(Debug, Serialize)]
struct JudgeChatRequest<'a> {
    model: &'a str,
    messages: Vec<JudgeChatMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    response_format: JudgeResponseFormat,
}

#[derive(Debug, Serialize)]
struct JudgeChatMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct JudgeResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JudgeJsonSchema,
}

#[derive(Debug, Serialize)]
struct JudgeJsonSchema {
    description: Option<String>,
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
struct JudgeChatResponse {
    choices: Vec<JudgeChatChoice>,
}

#[derive(Debug, Deserialize)]
struct JudgeChatChoice {
    finish_reason: Option<String>,
    message: JudgeChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct JudgeChatResponseMessage {
    content: Option<String>,
    refusal: Option<Value>,
}

pub fn decrypt_xor_base64(ciphertext_b64: &str, password: &str) -> Result<String, String> {
    let ciphertext = BASE64_STANDARD
        .decode(ciphertext_b64.trim())
        .map_err(|err| format!("base64 decode failed: {err}"))?;
    let key = derive_repeated_sha256_key(password, ciphertext.len());
    let plaintext = ciphertext
        .iter()
        .zip(key.iter())
        .map(|(lhs, rhs)| lhs ^ rhs)
        .collect::<Vec<_>>();
    String::from_utf8(plaintext).map_err(|err| format!("utf8 decode failed: {err}"))
}

pub fn build_browsecomp_expected_context(system_prompt: &str, user_prompt: &str) -> String {
    get_expected_context(system_prompt, user_prompt, &[])
}

pub fn build_browsecomp_turn_completion_prompt(cot_context: &str, cot: &str) -> String {
    build_turn_completion_prompt(cot_context, cot)
}

fn build_browsecomp_cot_prompt(expected_context: &str) -> String {
    expected_context
        .split_once("<|completions_of_cot|>")
        .map(|(prefix, _)| prefix.to_string())
        .unwrap_or_else(|| expected_context.to_string())
}

pub async fn generate_browsecomp_answer(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    expected_context: &str,
    sampling_config: &SamplingConfig,
) -> (String, String) {
    let cot_prompt = build_browsecomp_cot_prompt(expected_context);
    let mut cot_stop = vec!["</think>".to_string()];
    cot_stop.extend(
        BROWSECOMP_CONTROL_MARKERS
            .iter()
            .map(|marker| (*marker).to_string()),
    );
    let cot = get_completion(
        model_client,
        model_name,
        &cot_prompt,
        sampling_config,
        cot_stop,
        2048,
    )
    .await;
    let answer_prompt = build_browsecomp_turn_completion_prompt(expected_context, &cot);
    let mut answer_stop = vec!["\nUser:".to_string(), "\nAssistant:".to_string()];
    answer_stop.extend(
        BROWSECOMP_CONTROL_MARKERS
            .iter()
            .map(|marker| (*marker).to_string()),
    );
    let answer = get_completion(
        model_client,
        model_name,
        &answer_prompt,
        sampling_config,
        answer_stop,
        256,
    )
    .await
    .trim()
    .to_string();

    (format!("{answer_prompt}{answer}"), answer)
}

pub async fn judge_with_retry(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    locale: BrowseCompLocale,
    question: &str,
    model_response: &str,
    correct_answer: &str,
) -> BrowseJudgeOutcome {
    for attempt in 1..=3 {
        match judge_once(
            judger_client,
            judger_model_name,
            locale,
            question,
            model_response,
            correct_answer,
        )
        .await
        {
            Ok(outcome) => return outcome,
            Err(err) if attempt < 3 => {
                eprintln!(
                    "browsecomp llm judge failed on attempt {attempt}/3, retrying: {err}; model={judger_model_name}"
                );
            }
            Err(err) => {
                eprintln!(
                    "browsecomp llm judge failed after 3 attempts, marking sample failed with reason=0: {err}; model={judger_model_name}"
                );
                return BrowseJudgeOutcome {
                    is_passed: false,
                    reason: "0".to_string(),
                };
            }
        }
    }

    BrowseJudgeOutcome {
        is_passed: false,
        reason: "0".to_string(),
    }
}

fn derive_repeated_sha256_key(password: &str, length: usize) -> Vec<u8> {
    let digest = Sha256::digest(password.as_bytes()).to_vec();
    digest
        .iter()
        .copied()
        .cycle()
        .take(length)
        .collect::<Vec<_>>()
}

async fn judge_once(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    locale: BrowseCompLocale,
    question: &str,
    model_response: &str,
    correct_answer: &str,
) -> Result<BrowseJudgeOutcome, String> {
    let prompt = match locale {
        BrowseCompLocale::En => format!(
            concat!(
                "You are a rigorous answer judge.\n",
                "Decide whether the response correctly answers the question according to the correct answer.\n",
                "Treat small numerical tolerance as acceptable.\n",
                "Return only JSON matching the provided schema.\n\n",
                "[question]\n{question}\n\n",
                "[response]\n{model_response}\n\n",
                "[correct_answer]\n{correct_answer}\n"
            ),
            question = question,
            model_response = model_response,
            correct_answer = correct_answer,
        ),
        BrowseCompLocale::Zh => format!(
            concat!(
                "你是严格的答案判定器。\n",
                "请根据 correct_answer 判断 response 是否正确回答了 question。\n",
                "数值题可接受很小的误差。\n",
                "只返回符合给定 schema 的 JSON。\n\n",
                "[question]\n{question}\n\n",
                "[response]\n{model_response}\n\n",
                "[correct_answer]\n{correct_answer}\n"
            ),
            question = question,
            model_response = model_response,
            correct_answer = correct_answer,
        ),
    };

    let req = JudgeChatRequest {
        model: judger_model_name,
        messages: vec![JudgeChatMessage {
            role: "user",
            content: &prompt,
        }],
        temperature: BROWSECOMP_JUDGE_SAMPLING_CONFIG.temperature,
        top_p: BROWSECOMP_JUDGE_SAMPLING_CONFIG.top_p,
        max_completion_tokens: 128,
        response_format: JudgeResponseFormat {
            kind: "json_schema",
            json_schema: JudgeJsonSchema {
                description: None,
                name: "browsecomp_judger".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "is_passed": { "type": "boolean" },
                        "reason": { "type": "string" }
                    },
                    "required": ["is_passed", "reason"],
                    "additionalProperties": false
                }),
                strict: true,
            },
        },
    };

    let resp: JudgeChatResponse = judger_client
        .chat()
        .create_byot(&req)
        .await
        .map_err(|err| format!("judge request failed: {err}"))?;
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
    let wire = sonic_rs::from_str::<BrowseJudgeWire>(&content)
        .map_err(|err| format!("invalid judge json: {err}; content={content:?}"))?;

    Ok(BrowseJudgeOutcome {
        is_passed: wire.is_passed,
        reason: wire.reason.trim().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        BrowseCompLocale, build_browsecomp_cot_prompt, build_browsecomp_expected_context,
        build_browsecomp_turn_completion_prompt, decrypt_xor_base64,
    };
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
    use sha2::{Digest, Sha256};

    #[test]
    fn decrypts_xor_base64_payload() {
        let password = "canary";
        let plaintext = b"hello";
        let digest = Sha256::digest(password.as_bytes()).to_vec();
        let ciphertext = plaintext
            .iter()
            .zip(digest.iter().cycle())
            .map(|(lhs, rhs)| lhs ^ rhs)
            .collect::<Vec<_>>();
        let encoded = BASE64_STANDARD.encode(ciphertext);
        assert_eq!(decrypt_xor_base64(&encoded, password).unwrap(), "hello");
    }

    #[test]
    fn builds_browsecomp_expected_context() {
        let text = build_browsecomp_expected_context("sys", "user");
        assert_eq!(
            text,
            "System: sys\n\nUser: user\n\nAssistant: <think><|completions_of_cot|>"
        );
    }

    #[test]
    fn builds_browsecomp_turn_completion_prompt() {
        let prompt = build_browsecomp_turn_completion_prompt(
            "Assistant: <think><|completions_of_cot|>",
            "x",
        );
        assert_eq!(prompt, "Assistant: <think>x</think>\n");
    }

    #[test]
    fn cot_prompt_excludes_placeholder_marker() {
        let prompt = build_browsecomp_cot_prompt(
            "System: sys\n\nUser: user\n\nAssistant: <think><|completions_of_cot|>",
        );
        assert_eq!(prompt, "System: sys\n\nUser: user\n\nAssistant: <think>");
    }

    #[test]
    fn locale_is_copyable() {
        let locale = BrowseCompLocale::En;
        assert_eq!(locale, BrowseCompLocale::En);
    }
}
