use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    Example, get_expect_context, get_final_answer_with_cot_mode, get_ref_answer,
};
use crate::datasets::utils::hf::downloader::download_hf_files;
use crate::datasets::utils::hf::viewer::get_split_row_count;
use crate::datasets::utils::parquet::{get_string, get_string_list, get_u8, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};

#[distributed_slice(ALL_BENCHMARKS)]
static MMLU_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmlu"),
    field: Field::Knowledge,
    display_name: "MMLU",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.5,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 1.0,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0, 5],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(Mmlu::new(dataset_root)),
};

pub struct Mmlu {
    dataset_root: PathBuf,
    dev: Vec<MmluItem>,
    test: Vec<MmluItem>,
}

pub struct MmluItem {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: u8,
}

impl Mmlu {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            dev: Vec::new(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Mmlu {
    fn load(&mut self) -> bool {
        self.dev.clear();
        self.test.clear();

        let dev_path = self
            .dataset_root
            .join("mmlu/all/dev-00000-of-00001.parquet");
        let test_path = self
            .dataset_root
            .join("mmlu/all/test-00000-of-00001.parquet");
        if !dev_path.is_file() || !test_path.is_file() {
            return true;
        }

        let parse_item = |row: &Row| MmluItem {
            question: get_string(row, "question"),
            subject: get_string(row, "subject"),
            choices: get_string_list(row, "choices"),
            answer: get_u8(row, "answer"),
        };
        self.dev = read_parquet_items(dev_path, parse_item);
        self.test = read_parquet_items(test_path, parse_item);

        self.dev.is_empty() || self.test.is_empty()
    }

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        let (remote_dev_len, remote_test_len) = runtime.block_on(async {
            tokio::join!(
                get_split_row_count("cais/mmlu", "all", "dev"),
                get_split_row_count("cais/mmlu", "all", "test"),
            )
        });

        self.dev.len() != remote_dev_len || self.test.len() != remote_test_len
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_hf_files(
            &self.dataset_root,
            "mmlu",
            "datasets/cais/mmlu",
            &[
                "all/dev-00000-of-00001.parquet",
                "all/test-00000-of-00001.parquet",
            ],
            8,
            "main",
        ));
        println!("mmlu dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        let item = &self.test[index];
        let few_shot_examples = self
            .dev
            .iter()
            .filter(|example| example.subject == item.subject)
            .take(n_shot as usize)
            .map(|example| Example {
                question: example.question.clone(),
                choices: example.choices.clone(),
                answer_index: example.answer,
            })
            .collect::<Vec<_>>();

        get_expect_context(
            &item.subject,
            &item.question,
            &item.choices,
            cot_mode,
            &few_shot_examples,
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        get_ref_answer(self.test[index].answer)
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        _judger_model_name: Option<&str>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> bool {
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);

        get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &item.choices,
            &expected_context,
            &MMLU_INFO.sampling_config,
            cot_mode,
        )
        .await
            == item.answer
    }
}
