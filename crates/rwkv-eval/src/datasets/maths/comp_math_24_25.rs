use crate::datasets::maths::{
    get_expect_context, get_final_answer_with_cot_mode, judge_with_retry,
};
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static COMP_MATH_24_25_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("comp_math_24_25"),
    field: Field::Maths,
    display_name: "Comp-Math-24-25",
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
    avg_ks: &[4],
    pass_ks: &[],
    with_llm_judger: true,
    create: |dataset_root| Box::new(CompMath2425::new(dataset_root)),
};

pub struct CompMath2425 {
    dataset_root: PathBuf,
    test: Vec<CompMath2425Item>,
}

pub struct CompMath2425Item {
    question: String,
    answer: String,
    subject: String,
}

impl CompMath2425 {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for CompMath2425 {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("comp_math_24_25")
            .join("comp-math-24-25_test.jsonl");
        if !path.is_file() {
            return true;
        }

        let as_text = |value: &Value| match value {
            Value::Null => None,
            Value::String(value) => Some(value.trim().to_string()),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            Value::Array(value) => serde_json::to_string(value).ok(),
            Value::Object(value) => serde_json::to_string(value).ok(),
        };
        let take = |row: &Map<String, Value>, keys: &[&str]| {
            keys.iter()
                .find_map(|key| row.get(*key).and_then(|value| as_text(value)))
        };

        self.test = read_jsonl_items::<Value, _>(&path)
            .into_iter()
            .filter_map(|row| {
                let row = row.as_object()?;
                let question = take(row, &["problem", "question"])?;
                let answer = take(row, &["expected_answer", "answer"]).unwrap_or_default();
                let subject = take(row, &["subset_for_metrics", "source", "subject"])
                    .unwrap_or_else(|| "math".to_string());
                Some((question, answer, subject))
            })
            .map(|(question, answer, subject)| CompMath2425Item {
                question,
                answer,
                subject,
            })
            .collect();

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.is_empty()
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            "comp_math_24_25",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("comp-math-24-25_test.jsonl"),
                url: "https://raw.githubusercontent.com/rwkv-rs/rwkv-skills/main/src/eval/datasets/data_prepper/free_answer/static/comp-math-24-25_test.jsonl".to_string(),
            }],
            1,
        ));
        println!("comp_math_24_25 dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.subject, &item.question, cot_mode, &[])
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
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> bool {
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let final_answer = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &COMP_MATH_24_25_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let judger_client = judger_client
            .unwrap_or_else(|| panic!("benchmark requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("benchmark requires judger_model_name but got None"));

        judge_with_retry(
            judger_client,
            judger_model_name,
            &expected_context,
            &self.test[index].answer,
            &final_answer,
        )
        .await
    }
}
