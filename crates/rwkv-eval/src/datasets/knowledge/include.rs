use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};

use crate::datasets::knowledge::{get_expect_context, get_final_answer_with_cot_mode, get_ref_answer};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::get_parquet_files;
use crate::datasets::utils::parquet::{get_string, get_u8, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};

const DATASET_ID: &str = "CohereLabs/include-base-44";
const LOCAL_ROOT_NAME: &str = "include";

#[distributed_slice(ALL_BENCHMARKS)]
static INCLUDE_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("include"),
    field: Field::Knowledge,
    display_name: "INCLUDE",
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
    create: |dataset_root| Box::new(Include::new(dataset_root)),
};

pub struct Include {
    dataset_root: PathBuf,
    validation: Vec<IncludeItem>,
    test: Vec<IncludeItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn downloads_and_reads_dataset() {
        let tempdir = tempfile::tempdir().unwrap();
        let mut benchmark = Include::new(tempdir.path());

        benchmark.download().await;

        assert!(!benchmark.load());
        assert!(!benchmark.check().await);
        assert!(!benchmark.get_ref_answer(0).trim().is_empty());
        assert!(
            !benchmark
                .get_expected_context(0, CoTMode::NoCoT, 0)
                .trim()
                .is_empty()
        );
    }
}

pub struct IncludeItem {
    language: String,
    domain: String,
    subject: String,
    question: String,
    choices: Vec<String>,
    answer_index: u8,
}

impl Include {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            validation: Vec::new(),
            test: Vec::new(),
        }
    }
}

fn language_and_split_from_path(path: &Path) -> (String, String) {
    let split = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("missing INCLUDE split dir for {}", path.display()))
        .to_string();
    let language = path
        .parent()
        .and_then(|parent| parent.parent())
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("missing INCLUDE language dir for {}", path.display()))
        .to_string();
    (language, split)
}

#[async_trait]
impl Benchmark for Include {
    fn load(&mut self) -> bool {
        self.validation.clear();
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        for path in parquet_paths {
            let (language, split) = language_and_split_from_path(&path);
            let items = read_parquet_items(&path, |row: &Row| IncludeItem {
                language: {
                    let row_language = get_string(row, "language");
                    if row_language.trim().is_empty() {
                        language.clone()
                    } else {
                        row_language
                    }
                },
                domain: get_string(row, "domain"),
                subject: get_string(row, "subject"),
                question: get_string(row, "question"),
                choices: vec![
                    get_string(row, "option_a"),
                    get_string(row, "option_b"),
                    get_string(row, "option_c"),
                    get_string(row, "option_d"),
                ],
                answer_index: get_u8(row, "answer"),
            });

            match split.as_str() {
                "validation" => self.validation.extend(items),
                "test" => self.test.extend(items),
                _ => panic!("unsupported INCLUDE split `{split}`"),
            }
        }

        self.validation.is_empty() || self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.validation.is_empty() || self.test.is_empty()
    }

    async fn download(&self) {
        let mut parquet_files = get_parquet_files(DATASET_ID)
            .await
            .into_iter()
            .filter(|file| file.filename.ends_with(".parquet"))
            .collect::<Vec<_>>();
        parquet_files.sort_unstable_by(|left, right| {
            (
                left.config.as_str(),
                left.split.as_str(),
                left.filename.as_str(),
            )
                .cmp(&(
                    right.config.as_str(),
                    right.split.as_str(),
                    right.filename.as_str(),
                ))
        });

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
        println!("include dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(n_shot, 0, "include only supports 0-shot");

        let item = &self.test[index];
        let subject = format!("{} / {} / {}", item.language, item.domain, item.subject);
        get_expect_context(&subject, &item.question, &item.choices, cot_mode, &[])
    }

    fn get_ref_answer(&self, index: usize) -> String {
        get_ref_answer(self.test[index].answer_index)
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
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &item.choices,
            &expected_context,
            &INCLUDE_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let ref_answer = self.get_ref_answer(index);
        let is_passed = generated.answer_index == item.answer_index;

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
