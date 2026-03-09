use std::path::{Path, PathBuf};

use parquet::record::Row;
use tokio::runtime::Runtime;

use crate::datasets::hf_downloader::download_hf_files;
use crate::datasets::hf_viewer::get_split_row_count;
use crate::datasets::parquet_utils::{get_string, get_string_list, get_u8, read_parquet_items};
use crate::datasets::Benchmark;


pub struct Mmlu {
    dataset_root: PathBuf,
    is_checked: bool,
    dev: Vec<MmluItem>,
    test: Vec<MmluItem>,
}

struct MmluItem {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: u8,
}

impl Mmlu {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            is_checked: false,
            dev: Vec::new(),
            test: Vec::new(),
        }
    }
}

fn parse_item(row: &Row) -> MmluItem {
    MmluItem {
        question: get_string(row, "question"),
        subject: get_string(row, "subject"),
        choices: get_string_list(row, "choices"),
        answer: get_u8(row, "answer"),
    }
}

impl Benchmark for Mmlu {
    type Item = MmluItem;

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        let (remote_dev_len, remote_test_len) = runtime.block_on(async {
            tokio::join!(
                get_split_row_count("cais/mmlu", "all", "dev"),
                get_split_row_count("cais/mmlu", "all", "test"),
            )
        });

        self.dev.len() != remote_dev_len || self.test.len() != remote_test_len
    }
    
    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_hf_files(
            &self.dataset_root,
            "datasets/cais/mmlu",
            &[
                "all/dev-00000-of-00001.parquet",
                "all/test-00000-of-00001.parquet",
            ],
            8,
            "main",
        ));
        println!("mmlu dataset: {}", downloaded_path.display());
    }

    fn load(&mut self) {
        self.dev = read_parquet_items(self.dataset_root.join(
            "mmlu/all/dev-00000-of-00001.parquet"
        ), parse_item);
        self.test = read_parquet_items(self.dataset_root.join(
            "mmlu/all/test-00000-of-00001.parquet"
        ), parse_item);
    }

    fn get_expected_context(&self, item: Self::Item) -> String {
        let choices = item.choices.iter().enumerate()
            .map(|(i, choice)| {
                format!("{}. {}", char::from(b'A' + i as u8), choice)
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            concat!(
                "User: You are a very talented expert in {subject}.\n",
                "Answer this question and finish with a single option letter.\n",
                "Question: {question}\n",
                "Choices:\n{choices}\n",
                "Assistant: Therefore, the answer is"
            ),
            subject = item.subject,
            question = item.question,
            choices = choices,
        )
    }

    fn get_ref_answer(&self, item: Self::Item) -> String {
        let answer = item.answer;
        char::from(b'A' + answer).to_string()
    }

    fn get_evaluator(&self) {
        
    }
}
