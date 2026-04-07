use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use sonic_rs::{Object as Map, Value, prelude::*};

use crate::cores::datasets::{
    ALL_BENCHMARKS,
    Benchmark,
    BenchmarkInfo,
    BenchmarkName,
    CoTMode,
    Field,
    Record,
    SamplingConfig,
    maths::{get_expect_context, get_final_answer_with_cot_mode, judge_with_retry},
    utils::{
        hf::downloader::{UrlDownloadFile, download_url_files},
        jsonl::read_jsonl_items,
    },
};

#[distributed_slice(ALL_BENCHMARKS)]
static GSM8K_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("gsm8k"),
    field: Field::Maths,
    display_name: "GSM8K",
    cot_mode: &[CoTMode::CoT],
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
    create: |dataset_root| Box::new(Gsm8k::new(dataset_root)),
};

pub struct Gsm8k {
    dataset_root: PathBuf,
    test: Vec<Gsm8kItem>,
}

pub struct Gsm8kItem {
    question: String,
    answer: String,
    subject: String,
}

impl Gsm8k {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn parse_gsm8k_answer(answer: &str) -> String {
    let final_part = answer
        .rsplit("####")
        .next()
        .unwrap_or(answer)
        .trim()
        .replace(',', "");
    find_last_number(&final_part).unwrap_or(final_part)
}

fn find_last_number(text: &str) -> Option<String> {
    let mut best = None;
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | ',') {
            current.push(ch);
            continue;
        }
        if !current.is_empty() {
            let value = current.trim_matches(',').replace(',', "");
            if !value.is_empty() {
                best = Some(value);
            }
            current.clear();
        }
    }

    if !current.is_empty() {
        let value = current.trim_matches(',').replace(',', "");
        if !value.is_empty() {
            best = Some(value);
        }
    }

    best
}

#[async_trait]
impl Benchmark for Gsm8k {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self.dataset_root.join("gsm8k").join("test.jsonl");
        if !path.is_file() {
            return true;
        }

        let take = |row: &Map, keys: &[&str]| {
            keys.iter().find_map(|key| {
                row.get(&key)
                    .and_then(crate::cores::datasets::maths::json_value_as_text)
            })
        };

        self.test = read_jsonl_items::<Value, _>(&path)
            .into_iter()
            .filter_map(|row| {
                let row = row.as_object()?;
                let question = take(row, &["question", "problem"])?;
                let answer = parse_gsm8k_answer(&take(row, &["answer"]).unwrap_or_default());
                Some((question, answer, "gsm8k".to_string()))
            })
            .map(|(question, answer, subject)| Gsm8kItem {
                question,
                answer,
                subject,
            })
            .collect();

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "gsm8k",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("test.jsonl"),
                url: "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl".to_string(),
            }],
            1,
        ).await;
        println!("gsm8k dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.subject, &item.question, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].answer.clone()
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        judger_model_name: Option<&str>,
        judger_client: Option<&Client<OpenAIConfig>>,
        _sandbox_queue: &crate::cores::sandbox_queue::SandboxQueue,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &GSM8K_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let judger_client = judger_client
            .unwrap_or_else(|| panic!("benchmark requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("benchmark requires judger_model_name but got None"));

        let ref_answer = self.get_ref_answer(index);
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            &expected_context,
            &ref_answer,
            &generated.answer,
        )
        .await;

        Record {
            context: generated.context,
            answer: generated.answer,
            ref_answer,
            is_passed: outcome.is_passed,
            fail_reason: outcome.fail_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GSM8K_INFO;
    crate::cores::datasets::benchmark_dataset_tests!(GSM8K_INFO);
}
