use async_openai::Client;
use async_openai::config::OpenAIConfig;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    get_expected_context, get_ref_answer, judge_multiple_choice,
};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::get_parquet_files;
use crate::datasets::utils::parquet::{get_i64, get_string, get_string_list, read_parquet_items};

const DATASET_ID: &str = "TIGER-Lab/MMLU-Pro";
const LOCAL_ROOT_NAME: &str = "mmlu_pro";

#[distributed_slice(ALL_BENCHMARKS)]
static MMLU_PRO_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmlu_pro"),
    field: Field::Knowledge,
    display_name: "MMLU-Pro",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.5,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 1.0,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    avg_ks: &[1],
    pass_ks: &[1],
    with_llm_judger: false,
};

pub struct MmluPro {
    dataset_root: PathBuf,
    validation: Vec<MmluProItem>,
    test: Vec<MmluProItem>,
}

#[allow(dead_code)]
pub struct MmluProItem {
    question_id: i64,
    question: String,
    options: Vec<String>,
    answer: String,
    answer_index: i64,
    cot_content: String,
    category: String,
    src: String,
}

impl MmluPro {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            validation: Vec::new(),
            test: Vec::new(),
        }
    }
}

impl Benchmark for MmluPro {
    type Item = MmluProItem;

    fn load(&mut self) {
        self.validation.clear();
        self.test.clear();

        for path in collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet")
        {
            let split = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap()
                .to_string();

            let items = read_parquet_items(&path, |row: &Row| MmluProItem {
                question_id: get_i64(row, "question_id"),
                question: get_string(row, "question"),
                options: get_string_list(row, "options"),
                answer: get_string(row, "answer"),
                answer_index: get_i64(row, "answer_index"),
                cot_content: get_string(row, "cot_content"),
                category: get_string(row, "category"),
                src: get_string(row, "src"),
            });

            match split.as_str() {
                "validation" => self.validation.extend(items),
                "test" => self.test.extend(items),
                _ => {}
            }
        }
    }

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        let dataset_root = self.dataset_root.join(LOCAL_ROOT_NAME);

        runtime
            .block_on(get_parquet_files(DATASET_ID))
            .into_iter()
            .filter(|file| file.config == "default")
            .any(|file| !dataset_root.join(file.relative_path()).exists())
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let parquet_files = runtime
            .block_on(get_parquet_files(DATASET_ID))
            .into_iter()
            .filter(|file| file.config == "default")
            .collect::<Vec<_>>();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            LOCAL_ROOT_NAME,
            &parquet_files
                .into_iter()
                .map(|file| UrlDownloadFile {
                    relative_path: file.relative_path(),
                    url: file.url,
                })
                .collect::<Vec<_>>(),
            4,
        ));
        println!("mmlu_pro dataset: {}", downloaded_path.display());
    }

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        get_expected_context(&item.category, &item.question, &item.options, cot_mode)
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        get_ref_answer(&u8::try_from(item.answer_index).unwrap())
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &Client<OpenAIConfig>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool {
        let expected_context = self.get_expected_context(item, cot_mode);
        let answer = u8::try_from(item.answer_index).unwrap();

        judge_multiple_choice(
            model_client,
            &model_name,
            &item.options,
            &expected_context,
            &MMLU_PRO_INFO.sampling_config,
            cot_mode,
            answer,
        )
        .await
    }
}
