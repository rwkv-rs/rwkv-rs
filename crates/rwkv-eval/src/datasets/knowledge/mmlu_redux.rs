use async_openai::Client;
use async_openai::config::OpenAIConfig;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    get_ref_answer, get_final_answer_with_cot_mode, get_expect_context,
};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::get_parquet_files;
use crate::datasets::utils::parquet::{
    get_i64, get_optional_string, get_string, get_string_list, read_parquet_items,
};

const DATASET_ID: &str = "edinburgh-dawg/mmlu-redux-2.0";
const LOCAL_ROOT_NAME: &str = "mmlu_redux";

#[distributed_slice(ALL_BENCHMARKS)]
static MMLU_REDUX_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmlu_redux"),
    field: Field::Knowledge,
    display_name: "MMLU-Redux",
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

pub struct MmluRedux {
    dataset_root: PathBuf,
    test: Vec<MmluReduxItem>,
}

#[allow(dead_code)]
pub struct MmluReduxItem {
    question: String,
    choices: Vec<String>,
    answer: i64,
    error_type: String,
    source: Option<String>,
    correct_answer: Option<String>,
    potential_reason: Option<String>,
    config_name: String,
}

impl MmluRedux {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }

    fn resolved_answer_index(item: &MmluReduxItem) -> Option<u8> {
        match item.correct_answer.as_deref().map(str::trim) {
            Some("") => None,
            Some(correct_answer) => correct_answer
                .parse::<u8>()
                .ok()
                .filter(|idx| usize::from(*idx) < item.choices.len()),
            None => u8::try_from(item.answer)
                .ok()
                .filter(|idx| usize::from(*idx) < item.choices.len()),
        }
    }
}

impl Benchmark for MmluRedux {
    type Item = MmluReduxItem;

    fn load(&mut self) {
        self.test.clear();

        for path in collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet")
        {
            let config_name = path
                .parent()
                .and_then(|parent| parent.parent())
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap()
                .to_string();

            let items = read_parquet_items(&path, |row: &Row| MmluReduxItem {
                question: get_string(row, "question"),
                choices: get_string_list(row, "choices"),
                answer: get_i64(row, "answer"),
                error_type: get_string(row, "error_type"),
                source: get_optional_string(row, "source"),
                correct_answer: get_optional_string(row, "correct_answer"),
                potential_reason: get_optional_string(row, "potential_reason"),
                config_name: config_name.clone(),
            });

            self.test.extend(
                items
                    .into_iter()
                    .filter(|item| Self::resolved_answer_index(item).is_some()),
            );
        }
    }

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        let dataset_root = self.dataset_root.join(LOCAL_ROOT_NAME);

        runtime
            .block_on(get_parquet_files(DATASET_ID))
            .into_iter()
            .filter(|file| file.split == "test")
            .any(|file| !dataset_root.join(file.relative_path()).exists())
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let parquet_files = runtime
            .block_on(get_parquet_files(DATASET_ID))
            .into_iter()
            .filter(|file| file.split == "test")
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
            8,
        ));
        println!("mmlu_redux dataset: {}", downloaded_path.display());
    }

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        get_expect_context(&item.config_name, &item.question, &item.choices, cot_mode)
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        get_ref_answer(Self::resolved_answer_index(item).unwrap())
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &Client<OpenAIConfig>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool {
        let expected_context =
            get_expect_context(&item.config_name, &item.question, &item.choices, cot_mode);
        let answer_index = Self::resolved_answer_index(item).unwrap();

        get_final_answer_with_cot_mode(
            model_client,
            &model_name,
            &item.choices,
            &expected_context,
            &MMLU_REDUX_INFO.sampling_config,
            cot_mode,
        ).await == answer_index
    }
}
