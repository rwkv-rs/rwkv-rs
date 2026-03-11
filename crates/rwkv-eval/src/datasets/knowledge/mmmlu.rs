use async_openai::Client;
use async_openai::config::OpenAIConfig;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    answer_index_from_letter, get_ref_answer, get_final_answer_with_cot_mode,
    get_expect_context,
};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::get_parquet_files;
use crate::datasets::utils::parquet::{get_i64, get_string, read_parquet_items};

const DATASET_ID: &str = "openai/MMMLU";
const LOCAL_ROOT_NAME: &str = "mmmlu";

#[distributed_slice(ALL_BENCHMARKS)]
static MMMLU_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmmlu"),
    field: Field::Knowledge,
    display_name: "MMMLU",
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

pub struct Mmmlu {
    dataset_root: PathBuf,
    test: Vec<MmmluItem>,
}

#[allow(dead_code)]
pub struct MmmluItem {
    unnamed_0: i64,
    question: String,
    a: String,
    b: String,
    c: String,
    d: String,
    answer: String,
    subject: String,
}

impl Mmmlu {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

impl Benchmark for Mmmlu {
    type Item = MmmluItem;

    fn load(&mut self) {
        self.test.clear();

        for path in collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet")
        {
            let items = read_parquet_items(&path, |row: &Row| MmmluItem {
                unnamed_0: get_i64(row, "Unnamed: 0"),
                question: get_string(row, "Question"),
                a: get_string(row, "A"),
                b: get_string(row, "B"),
                c: get_string(row, "C"),
                d: get_string(row, "D"),
                answer: get_string(row, "Answer"),
                subject: get_string(row, "Subject"),
            });

            self.test.extend(items);
        }
    }

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        let dataset_root = self.dataset_root.join(LOCAL_ROOT_NAME);

        runtime
            .block_on(get_parquet_files(DATASET_ID))
            .into_iter()
            .filter(|file| file.config == "default" && file.split == "test")
            .any(|file| !dataset_root.join(file.relative_path()).exists())
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let parquet_files = runtime
            .block_on(get_parquet_files(DATASET_ID))
            .into_iter()
            .filter(|file| file.config == "default" && file.split == "test")
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
            2,
        ));
        println!("mmmlu dataset: {}", downloaded_path.display());
    }

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        get_expect_context(&item.subject, &item.question, &choices, cot_mode)
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
            get_expect_context(&item.subject, &item.question, &choices, cot_mode);
        let answer_index = answer_index_from_letter(&item.answer);

        get_final_answer_with_cot_mode(
            model_client,
            &model_name,
            &choices,
            &expected_context,
            &MMMLU_INFO.sampling_config,
            cot_mode,
        ).await == answer_index
    }
}
