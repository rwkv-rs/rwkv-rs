use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use sonic_rs::Value;

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
static AIME25_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("aime25"),
    field: Field::Maths,
    display_name: "AIME25",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.55,
        top_k: 66,
        top_p: 0.79,
        presence_penalty: 0.14,
        repetition_penalty: 0.01,
        penalty_decay: 0.997,
    },
    n_shots: &[0],
    avg_ks: &[64.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(Aime25::new(dataset_root)),
};

pub struct Aime25 {
    dataset_root: PathBuf,
    test: Vec<Aime25Item>,
}

#[derive(Clone, Deserialize)]
pub struct Aime25Item {
    id: String,
    problem: String,
    answer: Value,
}

impl Aime25 {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Aime25 {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self.dataset_root.join("aime25").join("test.jsonl");
        if !path.is_file() {
            return true;
        }

        self.test = read_jsonl_items::<Aime25Item, _>(&path);

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "aime25",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("test.jsonl"),
                url: "https://huggingface.co/datasets/math-ai/aime25/raw/main/test.jsonl"
                    .to_string(),
            }],
            1,
        )
        .await;
        println!("aime25 dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.problem, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        crate::cores::datasets::maths::json_value_as_text(&self.test[index].answer).unwrap_or_else(
            || {
                panic!(
                    "aime25 item has unsupported answer: {}",
                    self.test[index].id
                )
            },
        )
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
            &AIME25_INFO.sampling_config,
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
    use sonic_rs::json;

    use super::AIME25_INFO;

    #[test]
    fn raw_answer_numeric_to_text() {
        let answer = json!(70);
        let text = crate::cores::datasets::maths::json_value_as_text(&answer).unwrap();
        assert_eq!(text, "70");
    }

    crate::cores::datasets::benchmark_dataset_tests!(AIME25_INFO);
}
