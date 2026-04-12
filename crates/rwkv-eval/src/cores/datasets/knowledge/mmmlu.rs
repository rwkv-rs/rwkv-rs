use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;

use crate::cores::datasets::{
    ALL_BENCHMARKS,
    Benchmark,
    BenchmarkInfo,
    BenchmarkName,
    CoTMode,
    Field,
    Record,
    SamplingConfig,
    knowledge::{
        answer_index_from_letter,
        get_expect_context,
        get_final_answer_with_cot_mode,
        get_ref_answer,
    },
    utils::{
        collect_files_with_extension,
        hf::{
            downloader::{UrlDownloadFile, download_url_files},
            viewer::{get_parquet_files, get_split_row_count},
        },
        parquet::{get_i64, get_string, read_parquet_items},
    },
};

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
    n_shots: &[0],
    avg_ks: &[0.2],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(Mmmlu::new(dataset_root)),
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

#[async_trait]
impl Benchmark for Mmmlu {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        for path in parquet_paths {
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

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let remote_test_len = get_split_row_count(DATASET_ID, "default", "test").await;

        self.test.len() != remote_test_len
    }

    async fn download(&self) {
        let parquet_files = get_parquet_files(DATASET_ID)
            .await
            .into_iter()
            .filter(|file| file.config == "default" && file.split == "test")
            .collect::<Vec<_>>();
        let downloaded_path = download_url_files(
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
        )
        .await;
        println!("mmmlu dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        get_expect_context(&item.subject, &item.question, &choices, cot_mode, &[])
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
    ) -> Record {
        let item = &self.test[index];
        let choices = vec![
            item.a.clone(),
            item.b.clone(),
            item.c.clone(),
            item.d.clone(),
        ];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let answer_index = answer_index_from_letter(&item.answer);
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &choices,
            &expected_context,
            &MMMLU_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let ref_answer = self.get_ref_answer(index);
        let is_passed = generated.answer_index == answer_index;

        Record {
            context: generated.context,
            answer: generated.answer_text.clone(),
            ref_answer: ref_answer.clone(),
            is_passed,
            fail_reason: if is_passed {
                String::new()
            } else {
                format!(
                    "predicted {}, expected {}",
                    generated.answer_text, ref_answer
                )
            },
        }
    }
}

