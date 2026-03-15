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
use sonic_rs::{Object as Map, Value, prelude::*};
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static COLLEGE_MATH_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("college_math"),
    field: Field::Maths,
    display_name: "CollegeMath",
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
    avg_ks: &[8.0],
    pass_ks: &[8],
    with_llm_judger: true,
    create: |dataset_root| Box::new(CollegeMath::new(dataset_root)),
};

pub struct CollegeMath {
    dataset_root: PathBuf,
    test: Vec<CollegeMathItem>,
}

pub struct CollegeMathItem {
    question: String,
    answer: String,
    subject: String,
}

impl CollegeMath {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for CollegeMath {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self.dataset_root.join("college_math").join("test.jsonl");
        if !path.is_file() {
            return true;
        }

        let take = |row: &Map, keys: &[&str]| {
            keys.iter()
                .find_map(|key| row.get(&key).and_then(crate::datasets::maths::json_value_as_text))
        };

        self.test = read_jsonl_items::<Value, _>(&path)
            .into_iter()
            .filter_map(|row| {
                let row = row.as_object()?;
                let question = take(row, &["problem", "question"])?;
                let answer = take(row, &["expected_answer", "answer"]).unwrap_or_default();
                let subject = take(row, &["subject", "category", "domain", "topic", "source"])
                    .unwrap_or_else(|| "math".to_string());
                Some((question, answer, subject))
            })
            .map(|(question, answer, subject)| CollegeMathItem {
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
            "college_math",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("test.jsonl"),
                url: "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/college_math/test.jsonl".to_string(),
            }],
            1,
        ));
        println!("college_math dataset: {}", downloaded_path.display());
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
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> bool {
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let final_answer = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &COLLEGE_MATH_INFO.sampling_config,
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
