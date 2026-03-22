use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use crate::evaluators::instruction_following::{build_prompt, generate_response};
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
const ARENA_HARD_V2_SYSTEM_PROMPT: &str = concat!(
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. ",
    "You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n",
    "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n",
    "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\n",
    "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. ",
    "Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. ",
    "Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n",
    "Then consider the creativity and novelty of the assistant's answers when needed.\n\n",
    "Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n",
    "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n",
    "1. Assistant A is significantly better: [[A>>B]]\n",
    "2. Assistant A is slightly better: [[A>B]]\n",
    "3. Tie, relatively the same: [[A=B]]\n",
    "4. Assistant B is slightly better: [[B>A]]\n",
    "5. Assistant B is significantly better: [[B>>A]]\n\n",
    "Example output: \"My final verdict is tie: [[A=B]]\"."
);
const ARENA_HARD_V2_PROMPT_TEMPLATE: &str = concat!(
    "<|User Prompt|>\n{QUESTION}\n\n",
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n",
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"
);
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
    let judge_prompt = format!(
        "{}\n\n{}",
        ARENA_HARD_V2_SYSTEM_PROMPT,
        build_judge_user_prompt(question, answer_a, answer_b)
    );
    build_prompt(&judge_prompt)
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
        2048,
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
) -> ArenaJudgeOutcome {
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
            Ok(outcome) => return outcome,
            Err(err) if attempt < 3 => {
                eprintln!(
                    "arena-hard-v2 judge failed on attempt {attempt}/3, retrying: {err}; model={judger_model_name}"
                );
            }
            Err(err) => {
                panic!(
                    "arena-hard-v2 judge failed after 3 attempts: {err}; model={judger_model_name}; question={question}"
                );
            }
        }
    }

    panic!("unreachable arena-hard-v2 judge retry loop")
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
        build_prompt(&self.test[index].prompt)
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
        let answer = generate_response(
            model_client,
            model_name,
            &prompt,
            &ARENA_HARD_V2_INFO.sampling_config,
        )
        .await;
        let ref_answer = self.get_ref_answer(index);
        let context = format!(
            concat!(
                "[uid]\n{}\n\n",
                "[category]\n{}\n\n",
                "[subcategory]\n{}\n\n",
                "[language]\n{}\n\n",
                "[prompt]\n{}\n\n",
                "[model_answer]\n{}"
            ),
            item.uid,
            item.category,
            item.subcategory,
            item.language.clone().unwrap_or_default(),
            item.prompt,
            answer,
        );

        if answer.trim().is_empty() {
            return Record {
                context,
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

        let round_one_score = target_score_from_verdict(&round_one.verdict, false)
            .unwrap_or_else(|err| panic!("invalid Arena-Hard round one verdict: {err}"));
        let round_two_score = target_score_from_verdict(&round_two.verdict, true)
            .unwrap_or_else(|err| panic!("invalid Arena-Hard round two verdict: {err}"));
        let mean_score = (round_one_score + round_two_score) / 2.0;
        let is_passed = mean_score > 0.5;

        Record {
            context: format!(
                concat!(
                    "{}\n\n",
                    "[baseline_answer]\n{}\n\n",
                    "[judge_round_1]\nverdict={}\n{}\n\n",
                    "[judge_round_2]\nverdict={}\n{}"
                ),
                context,
                item.baseline_answer,
                round_one.verdict,
                round_one.reason,
                round_two.verdict,
                round_two.reason,
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

#[cfg(test)]
mod tests {
    use super::target_score_from_verdict;

    #[test]
    fn maps_verdicts_from_target_perspective() {
        assert_eq!(target_score_from_verdict("A>B", true).unwrap(), 1.0);
        assert_eq!(target_score_from_verdict("B>A", true).unwrap(), 0.0);
        assert_eq!(target_score_from_verdict("A=B", true).unwrap(), 0.5);
        assert_eq!(target_score_from_verdict("A>B", false).unwrap(), 0.0);
        assert_eq!(target_score_from_verdict("B>>A", false).unwrap(), 1.0);
    }
}
