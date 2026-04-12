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

const MAWPS_FILES: &[(&str, &str)] = &[
    (
        "addsub.jsonl",
        "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/addsub.jsonl",
    ),
    (
        "singleeq.jsonl",
        "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/singleeq.jsonl",
    ),
    (
        "singleop.jsonl",
        "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/singleop.jsonl",
    ),
    (
        "multiarith.jsonl",
        "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/multiarith.jsonl",
    ),
];

#[distributed_slice(ALL_BENCHMARKS)]
static MAWPS_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mawps"),
    field: Field::Maths,
    display_name: "MAWPS",
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
    avg_ks: &[2.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(Mawps::new(dataset_root)),
};

pub struct Mawps {
    dataset_root: PathBuf,
    test: Vec<MawpsItem>,
}

pub struct MawpsItem {
    question: String,
    answer: String,
    subject: String,
}

impl Mawps {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Mawps {
    fn load(&mut self) -> bool {
        self.test.clear();

        let take = |row: &Map, keys: &[&str]| {
            keys.iter().find_map(|key| {
                row.get(&key)
                    .and_then(crate::cores::datasets::maths::json_value_as_text)
            })
        };

        for (file_name, _) in MAWPS_FILES {
            let path = self.dataset_root.join("mawps").join(file_name);
            if !path.is_file() {
                return true;
            }

            let subject = file_name.trim_end_matches(".jsonl").to_string();
            self.test.extend(
                read_jsonl_items::<Value, _>(&path)
                    .into_iter()
                    .filter_map(|row| {
                        let row = row.as_object()?;
                        let question = take(row, &["problem", "input"])?;
                        let answer =
                            take(row, &["expected_answer", "target", "answer"]).unwrap_or_default();
                        let answer = answer
                            .parse::<f64>()
                            .ok()
                            .map(|value| {
                                if value.fract() == 0.0 {
                                    (value as i64).to_string()
                                } else {
                                    value.to_string()
                                }
                            })
                            .unwrap_or(answer);

                        Some((question, answer, subject.clone()))
                    })
                    .map(|(question, answer, subject)| MawpsItem {
                        question,
                        answer,
                        subject,
                    }),
            );
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let files = MAWPS_FILES
            .iter()
            .map(|(file_name, url)| UrlDownloadFile {
                relative_path: PathBuf::from(file_name),
                url: (*url).to_string(),
            })
            .collect::<Vec<_>>();
        let downloaded_path = download_url_files(&self.dataset_root, "mawps", &files, 4).await;
        println!("mawps dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.question, cot_mode)
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
            &MAWPS_INFO.sampling_config,
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
    use super::MAWPS_INFO;
    crate::cores::datasets::benchmark_dataset_tests!(MAWPS_INFO);
}
