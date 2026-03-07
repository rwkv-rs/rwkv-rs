use std::path::{Path, PathBuf};

use parquet::record::Row;
use tokio::runtime::Runtime;

use crate::datasets::hf_downloader::download_hf_files;
use crate::datasets::hf_viewer::get_split_row_count;
use crate::datasets::parquet_utils::{get_string, get_string_list, get_u8, read_parquet_items};
use crate::datasets::{Benchmark, BenchmarkSplit};
use crate::error::BenchmarkError;
use crate::evaluators::{Evaluator, MmluEvaluator};

const DATASET: &str = "cais/mmlu";
const DATASET_REPO: &str = "datasets/cais/mmlu";
const CONFIG: &str = "all";
const DEV_SPLIT: &str = "dev";
const TEST_SPLIT: &str = "test";
const AVG_K: usize = 1;
const PASS_K: usize = 1;
const WITH_LLM_JUDGER: bool = false;
const SPLITS: [BenchmarkSplit; 2] = [BenchmarkSplit::Dev, BenchmarkSplit::Test];

pub struct Mmlu {
    dataset_root: PathBuf,
    dev: MmluSplit,
    test: MmluSplit,
}

#[derive(Default)]
struct MmluSplit {
    questions: Vec<String>,
    subjects: Vec<String>,
    choices_list: Vec<Vec<String>>,
    answers: Vec<u8>,
}

struct MmluRowData {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: u8,
}

impl Mmlu {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            dev: MmluSplit::default(),
            test: MmluSplit::default(),
        }
    }

    fn repo_root(&self) -> PathBuf {
        self.dataset_root.join("mmlu")
    }

    fn split_root(&self) -> PathBuf {
        self.repo_root().join("all")
    }

    fn dev_file(&self) -> PathBuf {
        self.split_root().join("dev-00000-of-00001.parquet")
    }

    fn test_file(&self) -> PathBuf {
        self.split_root().join("test-00000-of-00001.parquet")
    }

    fn load_split(path: &Path) -> MmluSplit {
        let rows = read_parquet_items(path, parse_item);
        let mut split = MmluSplit::default();
        for row in rows {
            split.questions.push(row.question);
            split.subjects.push(row.subject);
            split.choices_list.push(row.choices);
            split.answers.push(row.answer);
        }
        split
    }

    fn split(&self, split: BenchmarkSplit) -> &MmluSplit {
        match split {
            BenchmarkSplit::Dev => &self.dev,
            BenchmarkSplit::Test => &self.test,
            _ => panic!("mmlu only supports dev/test splits"),
        }
    }
}

fn parse_item(row: &Row) -> MmluRowData {
    MmluRowData {
        question: get_string(row, "question"),
        subject: get_string(row, "subject"),
        choices: get_string_list(row, "choices"),
        answer: get_u8(row, "answer"),
    }
}

impl Benchmark for Mmlu {
    fn name(&self) -> &'static str {
        "mmlu"
    }

    fn dataset_root(&self) -> &Path {
        &self.dataset_root
    }

    fn dataset_dir(&self) -> PathBuf {
        self.repo_root()
    }

    fn avg_k(&self) -> usize {
        AVG_K
    }

    fn pass_k(&self) -> usize {
        PASS_K
    }

    fn with_llm_judger(&self) -> bool {
        WITH_LLM_JUDGER
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_hf_files(
            self.dataset_root(),
            DATASET_REPO,
            &[
                "all/dev-00000-of-00001.parquet",
                "all/test-00000-of-00001.parquet",
            ],
            8,
            "main",
        ));
        println!("mmlu dataset: {}", downloaded_path.display());
    }

    fn load(&mut self) -> Result<(), BenchmarkError> {
        if !self.repo_root().is_dir() {
            return Err(BenchmarkError::MissingDatasetDir(
                self.repo_root().display().to_string(),
            ));
        }
        if !self.dev_file().is_file() {
            return Err(BenchmarkError::MissingDatasetFile(
                self.dev_file().display().to_string(),
            ));
        }
        if !self.test_file().is_file() {
            return Err(BenchmarkError::MissingDatasetFile(
                self.test_file().display().to_string(),
            ));
        }

        self.dev = Self::load_split(&self.dev_file());
        self.test = Self::load_split(&self.test_file());
        Ok(())
    }

    fn check(&self) -> Result<(), BenchmarkError> {
        check_split("dev", &self.dev)?;
        check_split("test", &self.test)?;

        let runtime = Runtime::new().unwrap();
        let (remote_dev_len, remote_test_len) = runtime.block_on(async {
            tokio::join!(
                get_split_row_count(DATASET, CONFIG, DEV_SPLIT),
                get_split_row_count(DATASET, CONFIG, TEST_SPLIT),
            )
        });

        if self.dev.questions.len() != remote_dev_len
            || self.test.questions.len() != remote_test_len
        {
            return Err(BenchmarkError::InvalidDataset(format!(
                "mmlu size mismatch, local(dev={}, test={}), remote(dev={}, test={})",
                self.dev.questions.len(),
                self.test.questions.len(),
                remote_dev_len,
                remote_test_len,
            )));
        }

        Ok(())
    }

    fn splits(&self) -> &'static [BenchmarkSplit] {
        &SPLITS
    }

    fn len(&self, split: BenchmarkSplit) -> usize {
        self.split(split).questions.len()
    }

    fn get_expected_context(&self, split: BenchmarkSplit, index: usize) -> String {
        let split = self.split(split);
        let question = &split.questions[index];
        let subject = &split.subjects[index];
        let choices = split.choices_list[index]
            .iter()
            .enumerate()
            .map(|(choice_index, choice)| {
                format!("{}. {}", char::from(b'A' + choice_index as u8), choice)
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            concat!(
                "User: You are a very talented expert in {subject}.\n",
                "Answer this question and finish with a single option letter.\n",
                "Question: {question}\n",
                "Choices:\n{choices}\n\n",
                "Assistant: Therefore, the answer is"
            ),
            subject = subject,
            question = question,
            choices = choices,
        )
    }

    fn get_ref_answer(&self, split: BenchmarkSplit, index: usize) -> String {
        let answer = self.split(split).answers[index];
        char::from(b'A' + answer).to_string()
    }

    fn get_evaluator(&self) -> Box<dyn Evaluator> {
        Box::new(MmluEvaluator)
    }
}

fn check_split(name: &str, split: &MmluSplit) -> Result<(), BenchmarkError> {
    let len = split.questions.len();
    if len == 0 {
        return Err(BenchmarkError::InvalidDataset(format!(
            "mmlu {name} split is empty"
        )));
    }

    if split.subjects.len() != len || split.choices_list.len() != len || split.answers.len() != len
    {
        return Err(BenchmarkError::InvalidDataset(format!(
            "mmlu {name} split columns have inconsistent lengths"
        )));
    }

    Ok(())
}
