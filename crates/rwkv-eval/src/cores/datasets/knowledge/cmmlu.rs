use std::{
    path::{Path, PathBuf},
    process::Command,
};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;

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
        Example,
        answer_index_from_letter,
        get_expect_context,
        get_final_answer_with_cot_mode,
        get_ref_answer,
    },
    utils::{
        collect_files_with_extension,
        csv::read_csv_items,
        hf::downloader::{UrlDownloadFile, download_url_files},
    },
};

const LOCAL_ROOT_NAME: &str = "cmmlu";
const ARCHIVE_NAME: &str = "cmmlu_v1_0_1.zip";
const ARCHIVE_URL: &str =
    "https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip";

#[distributed_slice(ALL_BENCHMARKS)]
static CMMLU_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("cmmlu"),
    field: Field::Knowledge,
    display_name: "CMMLU",
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
    create: |dataset_root| Box::new(Cmmlu::new(dataset_root)),
};

pub struct Cmmlu {
    dataset_root: PathBuf,
    dev: Vec<CmmluItem>,
    test: Vec<CmmluItem>,
}

#[derive(Debug, Deserialize)]
struct CmmluCsvRow {
    #[serde(rename = "")]
    row_id: Option<i64>,
    #[serde(rename = "Question")]
    question: String,
    #[serde(rename = "A")]
    a: String,
    #[serde(rename = "B")]
    b: String,
    #[serde(rename = "C")]
    c: String,
    #[serde(rename = "D")]
    d: String,
    #[serde(rename = "Answer")]
    answer: String,
}

#[allow(dead_code)]
pub struct CmmluItem {
    row_id: Option<i64>,
    question: String,
    a: String,
    b: String,
    c: String,
    d: String,
    answer: String,
    subject_name: String,
}

impl Cmmlu {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            dev: Vec::new(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Cmmlu {
    fn load(&mut self) -> bool {
        self.dev.clear();
        self.test.clear();

        let root_dir = self.dataset_root.join(LOCAL_ROOT_NAME);
        if !root_dir.is_dir() {
            return true;
        }

        for split in ["dev", "test"] {
            let split_dir = root_dir.join(split);
            let csv_paths = collect_files_with_extension(&split_dir, "csv");
            if csv_paths.is_empty() {
                return true;
            }

            for path in csv_paths {
                let subject_name = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap()
                    .to_string();

                let rows = read_csv_items::<CmmluCsvRow, _>(&path)
                    .into_iter()
                    .map(|row| CmmluItem {
                        row_id: row.row_id,
                        question: row.question,
                        a: row.a,
                        b: row.b,
                        c: row.c,
                        d: row.d,
                        answer: row.answer,
                        subject_name: subject_name.clone(),
                    })
                    .collect::<Vec<_>>();

                match split {
                    "dev" => self.dev.extend(rows),
                    "test" => self.test.extend(rows),
                    _ => {}
                }
            }
        }

        self.dev.is_empty() || self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.dev.is_empty() || self.test.is_empty()
    }

    async fn download(&self) {
        let root_dir = download_url_files(
            &self.dataset_root,
            LOCAL_ROOT_NAME,
            &[UrlDownloadFile {
                relative_path: PathBuf::from(ARCHIVE_NAME),
                url: ARCHIVE_URL.to_string(),
            }],
            1,
        )
        .await;

        let zip_path = root_dir.join(ARCHIVE_NAME);
        let status = Command::new("unzip")
            .arg("-o")
            .arg(&zip_path)
            .arg("-d")
            .arg(&root_dir)
            .status()
            .unwrap_or_else(|e| {
                panic!("解压 CMMLU 文件失败: {}. error: {}", zip_path.display(), e)
            });
        if !status.success() {
            panic!(
                "解压 CMMLU 文件失败: zip={}, status={}",
                zip_path.display(),
                status
            );
        }

        println!("cmmlu dataset: {}", root_dir.display());
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
            .filter(|example| example.subject_name == item.subject_name)
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
            &item.subject_name,
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
        _sandbox_queue: &crate::cores::sandbox_queue::SandboxQueue,
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
            &CMMLU_INFO.sampling_config,
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
