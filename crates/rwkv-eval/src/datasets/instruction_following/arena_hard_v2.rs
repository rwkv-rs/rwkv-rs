use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    apply_user_assistant_template,
    instruction_following::sanitize_visible_answer,
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use crate::inferers::generate_text_completion;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use sonic_rs::{Value, prelude::*};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const ARENA_HARD_V2_EXPECTED_LEN: usize = 750;
const ARENA_HARD_V2_BASELINE_MODEL: &str = "o3-mini-2025-01-31";
const ARENA_HARD_V2_JUDGE_INSTRUCTION: &str = concat!(
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. ",
    "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n",
    "Do not generate a full replacement answer to the user prompt. Do not write corrected code, a rewritten essay, or any long standalone solution. ",
    "If you need a reference answer for comparison, keep it brief and internal to the evaluation.\n\n",
    "When evaluating the assistants' answers, compare both assistants' answers carefully and identify any mistakes or inaccurate information.\n\n",
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. ",
    "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. ",
    "Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n",
    "Then consider the creativity and novelty of the assistant's answers when needed.\n\n",
    "Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n",
    "Keep the evaluation concise. End with a final line that contains exactly one of the following verdict tags and nothing after it:\n\n",
    "1. Assistant A is significantly better: [[A>>B]]\n",
    "2. Assistant A is slightly better: [[A>B]]\n",
    "3. Tie, relatively the same: [[A=B]]\n",
    "4. Assistant B is slightly better: [[B>A]]\n",
    "5. Assistant B is significantly better: [[B>>A]]\n\n",
    "Example final line: [[A=B]]"
);
const ARENA_HARD_V2_PROMPT_TEMPLATE: &str = concat!(
    "<|User Prompt|>\n{QUESTION}\n\n",
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n",
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
);
const ARENA_HARD_V2_REQUEST_STOP_SUFFIXES: &[&str] = &["\n\nUser:"];
const ARENA_HARD_V2_SANITIZE_STOP_SUFFIXES: &[&str] = &["\n\nUser:", "\n\nAssistant:"];
const ARENA_HARD_V2_ASSISTANT_PREFIX: &str = "Here is my answer:\n";
const ARENA_HARD_V2_JUDGE_SAMPLING_CONFIG: SamplingConfig = SamplingConfig {
    temperature: 0.001,
    top_k: 1,
    top_p: 1.0,
    presence_penalty: 0.0,
    repetition_penalty: 0.0,
    penalty_decay: 1.0,
};

#[distributed_slice(ALL_BENCHMARKS)]
static ARENA_HARD_V2_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("arena_hard_v2"),
    field: Field::InstructionFollowing,
    display_name: "Arena-Hard v2",
    cot_mode: &[CoTMode::NoCoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 500,
        top_p: 0.4,
        presence_penalty: 0.5,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(ArenaHardV2::new(dataset_root)),
};

pub struct ArenaHardV2 {
    dataset_root: PathBuf,
    test: Vec<ArenaHardV2Item>,
}

pub struct ArenaHardV2Item {
    uid: String,
    category: String,
    subcategory: String,
    language: Option<String>,
    prompt: String,
    baseline_answer: String,
}

#[derive(Debug, Deserialize)]
struct RawArenaQuestion {
    uid: String,
    category: String,
    subcategory: String,
    language: Option<String>,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct RawArenaAnswer {
    uid: String,
    messages: Vec<RawArenaMessage>,
}

#[derive(Debug, Deserialize)]
struct RawArenaMessage {
    role: String,
    content: Value,
}

#[derive(Debug)]
struct ArenaJudgeOutcome {
    verdict: String,
    reason: String,
}

impl ArenaHardV2 {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn arena_hard_v2_question_path(dataset_root: &Path) -> PathBuf {
    dataset_root.join("arena_hard_v2").join("question.jsonl")
}

fn arena_hard_v2_baseline_answer_path(dataset_root: &Path) -> PathBuf {
    dataset_root
        .join("arena_hard_v2")
        .join("model_answer")
        .join(format!("{ARENA_HARD_V2_BASELINE_MODEL}.jsonl"))
}

fn value_as_text(value: &Value) -> Option<String> {
    if let Some(value) = value.as_str() {
        Some(value.trim().to_string())
    } else if value.as_number().is_some() || value.as_bool().is_some() {
        Some(value.to_string())
    } else {
        None
    }
}

fn extract_answer_text(messages: &[RawArenaMessage]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        if message.role != "assistant" {
            return None;
        }

        message
            .content
            .get("answer")
            .and_then(value_as_text)
            .or_else(|| value_as_text(&message.content))
            .map(|answer| answer.trim().to_string())
            .filter(|answer| !answer.is_empty())
    })
}

fn build_judge_user_prompt(question: &str, answer_a: &str, answer_b: &str) -> String {
    ARENA_HARD_V2_PROMPT_TEMPLATE
        .replace("{QUESTION}", question)
        .replace("{ANSWER_A}", answer_a)
        .replace("{ANSWER_B}", answer_b)
}

fn build_judge_prompt(question: &str, answer_a: &str, answer_b: &str) -> String {
    format!(
        "{}\n\n{}",
        ARENA_HARD_V2_JUDGE_INSTRUCTION,
        build_judge_user_prompt(question, answer_a, answer_b)
    )
}

fn generation_stop_suffixes() -> Vec<String> {
    ARENA_HARD_V2_REQUEST_STOP_SUFFIXES
        .iter()
        .map(|suffix| (*suffix).to_string())
        .collect()
}

fn trim_transcript_contamination(text: &str) -> String {
    sanitize_visible_answer(text, ARENA_HARD_V2_SANITIZE_STOP_SUFFIXES)
}

fn extract_verdict(text: &str) -> Option<String> {
    for verdict in ["A>>B", "A>B", "A=B", "B>A", "B>>A"] {
        if text.contains(&format!("[[{verdict}]]")) {
            return Some(verdict.to_string());
        }
    }
    None
}

fn verdict_to_a_score(verdict: &str) -> Option<f64> {
    match verdict {
        "A>>B" | "A>B" => Some(1.0),
        "A=B" => Some(0.5),
        "B>A" | "B>>A" => Some(0.0),
        _ => None,
    }
}

fn target_score_from_verdict(verdict: &str, target_is_a: bool) -> Result<f64, String> {
    let a_score = verdict_to_a_score(verdict)
        .ok_or_else(|| format!("unsupported Arena-Hard verdict `{verdict}`"))?;
    Ok(if target_is_a { a_score } else { 1.0 - a_score })
}

async fn judge_once(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    question: &str,
    answer_a: &str,
    answer_b: &str,
) -> Result<ArenaJudgeOutcome, String> {
    let prompt = build_judge_prompt(question, answer_a, answer_b);
    let content = generate_text_completion(
        judger_client,
        judger_model_name,
        &prompt,
        vec![],
        512,
        &ARENA_HARD_V2_JUDGE_SAMPLING_CONFIG,
    )
    .await
    .map_err(|err| format!("judge request failed: {err}"))?;
    let verdict = extract_verdict(&content)
        .ok_or_else(|| format!("judge returned no Arena-Hard verdict tag: {content:?}"))?;

    Ok(ArenaJudgeOutcome {
        verdict,
        reason: content.trim().to_string(),
    })
}

async fn judge_with_retry(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    question: &str,
    answer_a: &str,
    answer_b: &str,
) -> Result<ArenaJudgeOutcome, String> {
    for attempt in 1..=3 {
        match judge_once(
            judger_client,
            judger_model_name,
            question,
            answer_a,
            answer_b,
        )
        .await
        {
            Ok(outcome) => return Ok(outcome),
            Err(err) if attempt < 3 => {
                eprintln!(
                    "arena-hard-v2 judge failed on attempt {attempt}/3, retrying: {err}; model={judger_model_name}"
                );
            }
            Err(err) => {
                return Err(format!(
                    "arena-hard-v2 judge failed after 3 attempts: {err}; model={judger_model_name}; question={question}"
                ));
            }
        }
    }

    Err("unreachable arena-hard-v2 judge retry loop".to_string())
}

fn build_context_with_judges(
    base_context: &str,
    baseline_answer: &str,
    round_one: Option<&ArenaJudgeOutcome>,
    round_two: Option<&ArenaJudgeOutcome>,
    round_one_error: Option<&str>,
    round_two_error: Option<&str>,
) -> String {
    let round_one_body = match (round_one, round_one_error) {
        (Some(outcome), _) => format!("verdict={}\n{}", outcome.verdict, outcome.reason),
        (None, Some(err)) => format!("error={err}"),
        (None, None) => "missing".to_string(),
    };
    let round_two_body = match (round_two, round_two_error) {
        (Some(outcome), _) => format!("verdict={}\n{}", outcome.verdict, outcome.reason),
        (None, Some(err)) => format!("error={err}"),
        (None, None) => "missing".to_string(),
    };

    format!(
        concat!(
            "{}\n\n",
            "[baseline_answer]\n{}\n\n",
            "[judge_round_1]\n{}\n\n",
            "[judge_round_2]\n{}"
        ),
        base_context, baseline_answer, round_one_body, round_two_body,
    )
}

fn build_judge_error_reason(round_one_error: Option<&str>, round_two_error: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(err) = round_one_error {
        parts.push(format!("round1={err}"));
    }
    if let Some(err) = round_two_error {
        parts.push(format!("round2={err}"));
    }

    if parts.is_empty() {
        "arena judge_error".to_string()
    } else {
        format!("arena judge_error: {}", parts.join(" | "))
    }
}

#[async_trait]
impl Benchmark for ArenaHardV2 {
    fn load(&mut self) -> bool {
        self.test.clear();

        let question_path = arena_hard_v2_question_path(&self.dataset_root);
        let baseline_path = arena_hard_v2_baseline_answer_path(&self.dataset_root);
        if !question_path.is_file() || !baseline_path.is_file() {
            return true;
        }

        let baseline_by_uid = read_jsonl_items::<RawArenaAnswer, _>(&baseline_path)
            .into_iter()
            .map(|row| {
                let answer = extract_answer_text(&row.messages).unwrap_or_else(|| {
                    panic!("failed to extract baseline answer for uid={}", row.uid)
                });
                (row.uid, answer)
            })
            .collect::<BTreeMap<_, _>>();

        self.test = read_jsonl_items::<RawArenaQuestion, _>(&question_path)
            .into_iter()
            .map(|row| {
                let baseline_answer = baseline_by_uid.get(&row.uid).cloned().unwrap_or_else(|| {
                    panic!("missing Arena-Hard baseline answer for uid={}", row.uid)
                });
                ArenaHardV2Item {
                    uid: row.uid,
                    category: row.category,
                    subcategory: row.subcategory,
                    language: row.language,
                    prompt: row.prompt,
                    baseline_answer,
                }
            })
            .collect();

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.len() != ARENA_HARD_V2_EXPECTED_LEN
            || self.test.iter().any(|item| {
                item.uid.trim().is_empty()
                    || item.category.trim().is_empty()
                    || item.subcategory.trim().is_empty()
                    || item.prompt.trim().is_empty()
                    || item.baseline_answer.trim().is_empty()
            })
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "arena_hard_v2",
            &[
                UrlDownloadFile {
                    relative_path: PathBuf::from("question.jsonl"),
                    url: "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/data/arena-hard-v2.0/question.jsonl".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from(format!(
                        "model_answer/{ARENA_HARD_V2_BASELINE_MODEL}.jsonl"
                    )),
                    url: format!(
                        "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/data/arena-hard-v2.0/model_answer/{ARENA_HARD_V2_BASELINE_MODEL}.jsonl"
                    ),
                },
            ],
            2,
        )
        .await;
        println!("arena_hard_v2 dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(
            cot_mode,
            CoTMode::NoCoT,
            "arena_hard_v2 only supports NoCoT"
        );
        assert_eq!(n_shot, 0, "arena_hard_v2 only supports 0-shot");
        apply_user_assistant_template(
            self.test[index].prompt.clone(),
            ARENA_HARD_V2_ASSISTANT_PREFIX.to_string(),
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].baseline_answer.clone()
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        judger_model_name: Option<&str>,
        judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let item = &self.test[index];
        let prompt = self.get_expected_context(index, cot_mode, n_shot);
        let raw_answer = generate_text_completion(
            model_client,
            model_name,
            &prompt,
            generation_stop_suffixes(),
            4096,
            &ARENA_HARD_V2_INFO.sampling_config,
        )
        .await
        .unwrap();
        let answer = trim_transcript_contamination(&raw_answer);
        let ref_answer = self.get_ref_answer(index);
        let base_context = format!(
            concat!(
                "[uid]\n{}\n\n",
                "[category]\n{}\n\n",
                "[subcategory]\n{}\n\n",
                "[language]\n{}\n\n",
                "[prompt]\n{}\n\n",
                "[raw_model_answer]\n{}\n\n",
                "[model_answer]\n{}"
            ),
            item.uid,
            item.category,
            item.subcategory,
            item.language.clone().unwrap_or_default(),
            item.prompt,
            raw_answer,
            answer,
        );

        if answer.trim().is_empty() {
            return Record {
                context: base_context,
                answer,
                ref_answer,
                is_passed: false,
                fail_reason: "model returned empty response".to_string(),
            };
        }

        let judger_client = judger_client
            .unwrap_or_else(|| panic!("arena_hard_v2 requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("arena_hard_v2 requires judger_model_name but got None"));

        let round_one = judge_with_retry(
            judger_client,
            judger_model_name,
            &item.prompt,
            &item.baseline_answer,
            &answer,
        )
        .await;
        let round_two = judge_with_retry(
            judger_client,
            judger_model_name,
            &item.prompt,
            &answer,
            &item.baseline_answer,
        )
        .await;

        if round_one.is_err() || round_two.is_err() {
            let round_one_error = round_one.as_ref().err().map(String::as_str);
            let round_two_error = round_two.as_ref().err().map(String::as_str);
            return Record {
                context: build_context_with_judges(
                    &base_context,
                    &item.baseline_answer,
                    round_one.as_ref().ok(),
                    round_two.as_ref().ok(),
                    round_one_error,
                    round_two_error,
                ),
                answer,
                ref_answer,
                is_passed: false,
                fail_reason: build_judge_error_reason(round_one_error, round_two_error),
            };
        }

        let round_one = round_one.unwrap();
        let round_two = round_two.unwrap();
        let round_one_score = match target_score_from_verdict(&round_one.verdict, false) {
            Ok(score) => score,
            Err(err) => {
                return Record {
                    context: build_context_with_judges(
                        &base_context,
                        &item.baseline_answer,
                        Some(&round_one),
                        Some(&round_two),
                        Some(err.as_str()),
                        None,
                    ),
                    answer,
                    ref_answer,
                    is_passed: false,
                    fail_reason: format!("arena judge_error: round1={err}"),
                };
            }
        };
        let round_two_score = match target_score_from_verdict(&round_two.verdict, true) {
            Ok(score) => score,
            Err(err) => {
                return Record {
                    context: build_context_with_judges(
                        &base_context,
                        &item.baseline_answer,
                        Some(&round_one),
                        Some(&round_two),
                        None,
                        Some(err.as_str()),
                    ),
                    answer,
                    ref_answer,
                    is_passed: false,
                    fail_reason: format!("arena judge_error: round2={err}"),
                };
            }
        };
        let mean_score = (round_one_score + round_two_score) / 2.0;
        let is_passed = mean_score > 0.5;

        Record {
            context: build_context_with_judges(
                &base_context,
                &item.baseline_answer,
                Some(&round_one),
                Some(&round_two),
                None,
                None,
            ),
            answer,
            ref_answer,
            is_passed,
            fail_reason: if is_passed {
                String::new()
            } else {
                format!(
                    "arena judge mean_score={mean_score:.2}; round1={}; round2={}",
                    round_one.verdict, round_two.verdict
                )
            },
        }
    }
}
