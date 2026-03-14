use crate::datasets::maths::{
    get_expect_context, get_final_answer_with_cot_mode, judge_with_retry,
};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::download_hf_parquet_splits;
use crate::datasets::utils::parquet::{get_i64, get_string, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static BEYOND_AIME_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("beyond_aime"),
    field: Field::Maths,
    display_name: "BeyondAIME",
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
    avg_ks: &[16.0],
    pass_ks: &[8],
    with_llm_judger: true,
    create: |dataset_root| Box::new(BeyondAime::new(dataset_root)),
};

pub struct BeyondAime {
    dataset_root: PathBuf,
    test: Vec<BeyondAimeItem>,
}

pub struct BeyondAimeItem {
    question: String,
    answer: String,
    subject: String,
}

impl BeyondAime {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for BeyondAime {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths = collect_files_with_extension(
            self.dataset_root.join("beyond_aime/default/test"),
            "parquet",
        );
        if parquet_paths.is_empty() {
            return true;
        }

        let parse_item = |row: &Row| BeyondAimeItem {
            question: get_string(row, "problem"),
            answer: get_i64(row, "answer").to_string(),
            subject: "math".to_string(),
        };
        for path in parquet_paths {
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.is_empty()
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_hf_parquet_splits(
            &self.dataset_root,
            "beyond_aime",
            "ByteDance-Seed/BeyondAIME",
            "default",
            &["test"],
            2,
        ));
        println!("beyond_aime dataset: {}", downloaded_path.display());
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
            &BEYOND_AIME_INFO.sampling_config,
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
