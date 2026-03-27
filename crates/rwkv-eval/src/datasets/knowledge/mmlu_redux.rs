use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;

use crate::datasets::{
    ALL_BENCHMARKS,
    Benchmark,
    BenchmarkInfo,
    BenchmarkName,
    CoTMode,
    Field,
    Record,
    SamplingConfig,
    knowledge::{get_expect_context, get_final_answer_with_cot_mode, get_ref_answer},
    utils::{
        collect_files_with_extension,
        hf::{
            downloader::{UrlDownloadFile, download_url_files},
            viewer::get_parquet_files,
        },
        parquet::{get_i64, get_optional_string, get_string, get_string_list, read_parquet_items},
    },
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
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(MmluRedux::new(dataset_root)),
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
}

#[async_trait]
impl Benchmark for MmluRedux {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        for path in parquet_paths {
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

            self.test.extend(items.into_iter().filter(|item| {
                match item.correct_answer.as_deref().map(str::trim) {
                    Some("") => false,
                    Some(correct_answer) => correct_answer
                        .parse::<u8>()
                        .ok()
                        .is_some_and(|idx| usize::from(idx) < item.choices.len()),
                    None => u8::try_from(item.answer)
                        .ok()
                        .is_some_and(|idx| usize::from(idx) < item.choices.len()),
                }
            }));
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let parquet_files = get_parquet_files(DATASET_ID)
            .await
            .into_iter()
            .filter(|file| file.split == "test")
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
            8,
        )
        .await;
        println!("mmlu_redux dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];
        get_expect_context(
            &item.config_name,
            &item.question,
            &item.choices,
            cot_mode,
            &[],
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        let item = &self.test[index];
        get_ref_answer(match item.correct_answer.as_deref().map(str::trim) {
            Some(correct_answer) if !correct_answer.is_empty() => {
                correct_answer.parse::<u8>().unwrap()
            }
            _ => u8::try_from(item.answer).unwrap(),
        })
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
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let answer_index = match item.correct_answer.as_deref().map(str::trim) {
            Some(correct_answer) if !correct_answer.is_empty() => {
                correct_answer.parse::<u8>().unwrap()
            }
            _ => u8::try_from(item.answer).unwrap(),
        };
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &item.choices,
            &expected_context,
            &MMLU_REDUX_INFO.sampling_config,
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
