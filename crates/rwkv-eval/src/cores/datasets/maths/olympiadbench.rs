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
static OLYMPIADBENCH_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("olympiadbench"),
    field: Field::Maths,
    display_name: "OlympiadBench",
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
    avg_ks: &[8.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(OlympiadBench::new(dataset_root)),
};

pub struct OlympiadBench {
    dataset_root: PathBuf,
    test: Vec<OlympiadBenchItem>,
}

#[derive(Debug, Deserialize)]
pub struct OlympiadBenchItem {
    id: u64,
    subfield: String,
    context: Option<String>,
    question: String,
    solution: Value,
    final_answer: Vec<String>,
    is_multiple_answer: bool,
    unit: Option<String>,
    answer_type: String,
    error: Option<String>,
}

impl OlympiadBench {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for OlympiadBench {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self.dataset_root.join("olympiadbench").join("test.jsonl");
        if !path.is_file() {
            return true;
        }

        self.test = read_jsonl_items::<OlympiadBenchItem, _>(&path);

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "olympiadbench",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("test.jsonl"),
                url: "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/olympiadbench/test.jsonl".to_string(),
            }],
            1,
        ).await;
        println!("olympiadbench dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.question, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index]
            .final_answer
            .first()
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "olympiadbench row missing final_answer at index={index}; id={}",
                    self.test[index].id
                )
            })
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
            &OLYMPIADBENCH_INFO.sampling_config,
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
