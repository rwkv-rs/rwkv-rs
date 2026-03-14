use crate::datasets::maths::{
    get_expect_context, get_final_answer_with_cot_mode, judge_with_retry,
};
use crate::datasets::utils::hf::downloader::download_hf_files;
use crate::datasets::utils::parquet::{get_optional_string_list, get_string, read_parquet_items};
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
static HMMT_FEB25_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("hmmt_feb25"),
    field: Field::Maths,
    display_name: "HMMT-Feb25",
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
    create: |dataset_root| Box::new(HmmtFeb25::new(dataset_root)),
};

pub struct HmmtFeb25 {
    dataset_root: PathBuf,
    test: Vec<HmmtFeb25Item>,
}

pub struct HmmtFeb25Item {
    question: String,
    answer: String,
    subject: String,
}

impl HmmtFeb25 {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for HmmtFeb25 {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("hmmt_feb25/default/train/0000.parquet");
        if !path.is_file() {
            return true;
        }

        let parse_item = |row: &Row| HmmtFeb25Item {
            question: get_string(row, "problem"),
            answer: get_string(row, "answer"),
            subject: get_optional_string_list(row, "problem_type")
                .and_then(|values| values.into_iter().next())
                .unwrap_or_else(|| "math".to_string()),
        };
        self.test = read_parquet_items(path, parse_item);

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.is_empty()
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_hf_files(
            &self.dataset_root,
            "hmmt_feb25",
            "datasets/MathArena/hmmt_feb_2025",
            &["default/train/0000.parquet"],
            1,
            "refs/convert/parquet",
        ));
        println!("hmmt_feb25 dataset: {}", downloaded_path.display());
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
            &HMMT_FEB25_INFO.sampling_config,
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
