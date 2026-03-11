use async_openai::Client;
use async_openai::config::OpenAIConfig;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    answer_index_from_letter, get_expect_context, get_final_answer_with_cot_mode, get_ref_answer,
};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::get_parquet_files;
use crate::datasets::utils::parquet::{get_i64, get_string, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};

const DATASET_ID: &str = "ceval/ceval-exam";
const LOCAL_ROOT_NAME: &str = "ceval";

#[distributed_slice(ALL_BENCHMARKS)]
static CEVAL_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("ceval"),
    field: Field::Knowledge,
    display_name: "C-Eval",
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

pub struct Ceval {
    dataset_root: PathBuf,
    dev: Vec<CevalItem>,
    validation: Vec<CevalItem>,
    test: Vec<CevalItem>,
}

#[allow(dead_code)]
pub struct CevalItem {
    id: i64,
    question: String,
    a: String,
    b: String,
    c: String,
    d: String,
    answer: String,
    explanation: String,
    config_name: String,
}

impl Ceval {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            dev: Vec::new(),
            validation: Vec::new(),
            test: Vec::new(),
        }
    }
}

impl Benchmark for Ceval {
    type Item = CevalItem;

    fn load(&mut self) {
        self.dev.clear();
        self.validation.clear();
        self.test.clear();

        for path in collect_files_with_extension(
            self.dataset_root.join(LOCAL_ROOT_NAME),
            "parquet",
        ) {
            let split = path.parent().and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str()).unwrap().to_string();
            let config_name = path.parent()
                .and_then(|parent| parent.parent())
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str()).unwrap().to_string();

            let items = read_parquet_items(&path, |row: &Row| CevalItem {
                id: get_i64(row, "id"),
                question: get_string(row, "question"),
                a: get_string(row, "A"),
                b: get_string(row, "B"),
                c: get_string(row, "C"),
                d: get_string(row, "D"),
                answer: get_string(row, "answer"),
                explanation: get_string(row, "explanation"),
                config_name: config_name.clone(),
            });

            match split.as_str() {
                "dev" => self.dev.extend(items),
                "val" => self.validation.extend(items),
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
            .any(|file| !dataset_root.join(file.relative_path()).exists())
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let parquet_files = runtime.block_on(get_parquet_files(DATASET_ID));
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
        println!("ceval dataset: {}", downloaded_path.display());
    }

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        get_expect_context(&item.config_name, &item.question, &choices, cot_mode)
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        get_ref_answer(answer_index_from_letter(&item.answer))
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &Client<OpenAIConfig>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool {
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        let expected_context =
            get_expect_context(&item.config_name, &item.question, &choices, cot_mode);
        let answer_index = answer_index_from_letter(&item.answer);

        get_final_answer_with_cot_mode(
            model_client,
            &model_name,
            &choices,
            &expected_context,
            &CEVAL_INFO.sampling_config,
            cot_mode,
        ).await == answer_index
    }
}
