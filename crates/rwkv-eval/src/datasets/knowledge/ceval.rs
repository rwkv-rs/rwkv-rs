use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    Example, answer_index_from_letter, get_expect_context, get_final_answer_with_cot_mode,
    get_ref_answer,
};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::{get_parquet_files, get_split_row_count};
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
    n_shots: &[0, 5],
    avg_ks: &[1],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(Ceval::new(dataset_root)),
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

#[async_trait]
impl Benchmark for Ceval {
    fn load(&mut self) -> bool {
        self.dev.clear();
        self.validation.clear();
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        for path in parquet_paths {
            let split = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap()
                .to_string();
            let config_name = path
                .parent()
                .and_then(|parent| parent.parent())
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap()
                .to_string();

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

        self.dev.is_empty() || self.validation.is_empty() || self.test.is_empty()
    }

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        let (remote_dev_len, remote_validation_len, remote_test_len) = runtime.block_on(async {
            let mut splits = BTreeSet::new();
            for file in get_parquet_files(DATASET_ID).await {
                splits.insert((file.config, file.split));
            }

            let mut dev_len = 0;
            let mut validation_len = 0;
            let mut test_len = 0;

            for (config, split) in splits {
                let split_len = get_split_row_count(DATASET_ID, &config, &split).await;
                match split.as_str() {
                    "dev" => dev_len += split_len,
                    "val" => validation_len += split_len,
                    "test" => test_len += split_len,
                    _ => {}
                }
            }

            (dev_len, validation_len, test_len)
        });

        self.dev.len() != remote_dev_len
            || self.validation.len() != remote_validation_len
            || self.test.len() != remote_test_len
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

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        let item = &self.test[index];
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        let few_shot_examples = self
            .dev
            .iter()
            .filter(|example| example.config_name == item.config_name)
            .take(n_shot as usize)
            .map(|example| Example {
                question: example.question.clone(),
                choices: vec![
                    example.a.clone(),
                    example.b.clone(),
                    example.c.clone(),
                    example.d.clone(),
                ],
                answer_index: answer_index_from_letter(&example.answer),
            })
            .collect::<Vec<_>>();

        get_expect_context(
            &item.config_name,
            &item.question,
            &choices,
            cot_mode,
            &few_shot_examples,
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        get_ref_answer(answer_index_from_letter(&self.test[index].answer))
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
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let answer_index = answer_index_from_letter(&item.answer);

        get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &choices,
            &expected_context,
            &CEVAL_INFO.sampling_config,
            cot_mode,
        )
        .await
            == answer_index
    }
}
