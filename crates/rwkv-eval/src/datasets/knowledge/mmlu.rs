use std::path::{Path, PathBuf};
use async_openai::types::completions::CreateCompletionRequest as BaseCompletionRequest;
use linkme::distributed_slice;
use parquet::record::Row;
use tokio::runtime::Runtime;

use crate::datasets::{Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, ALL_BENCHMARKS};
use crate::datasets::hf_downloader::download_hf_files;
use crate::datasets::hf_viewer::get_split_row_count;
use crate::datasets::parquet_utils::{get_string, get_string_list, get_u8, read_parquet_items};
use crate::runtime::OpenAiClient;

#[distributed_slice(ALL_BENCHMARKS)]
static MMLU_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmlu"),
    field: Field::Knowledge,
    display_name: "MMLU",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
    sampling_config: ,
    avg_ks: &[1],
    pass_ks: &[1],
    with_llm_judger: false,
};


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

impl Benchmark for Mmlu {
    type Item = MmluItem;

    fn load(&mut self) {
        let parse_item = |row: &Row| MmluItem {
            question: get_string(row, "question"),
            subject: get_string(row, "subject"),
            choices: get_string_list(row, "choices"),
            answer: get_u8(row, "answer"),
        };
        self.dev = read_parquet_items(
            self.dataset_root
                .join("mmlu/all/dev-00000-of-00001.parquet"),
            parse_item,
        );
        self.test = read_parquet_items(
            self.dataset_root
                .join("mmlu/all/test-00000-of-00001.parquet"),
            parse_item,
        );
    }

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

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        let choices = item
            .choices
            .iter()
            .enumerate()
            .map(|(i, choice)| format!("{}. {}", char::from(b'A' + i as u8), choice))
            .collect::<Vec<_>>()
            .join("\n");

        let user_part = format!(
            concat!(
                "User: You are a very talented expert in {subject}.\n",
                "Answer this question and finish with a single option letter.\n",
                "Question: {question}\n",
                "Choices:\n{choices}\n\n",
            ),
            subject = item.subject,
            question = item.question,
            choices = choices,
        );

        let assistant_part = match cot_mode {
            CoTMode::NoCoT => "Assistant: Therefore, the answer is <|logits|>",
            CoTMode::FakeCoT => "Assistant: <think>\n</think>\nTherefore, the answer is <|logits|>",
            CoTMode::CoT => concat!(
                "Assistant: <think<|completions_of_cot|></think>\n",
                "Therefore, the answer is <|logits_of_choices|>."
            ),
        };

        format!("User: {user_part}\n\nAssistant: {assistant_part}")
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        let answer = item.answer;
        char::from(b'A' + answer).to_string()
    }

    fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &OpenAiClient,
        judger_client: Option<&OpenAiClient>,
        cot_mode: CoTMode,
        fim_mode: bool,
        item: &Self::Item,
    ) -> bool {
        let expected_context = self.get_expected_context(item, cot_mode);
        if cot_mode == CoTMode::CoT {
            let prompt_for_cot = expected_context.split_once(
                "<|completions_of_cot|>"
            ).unwrap().0.to_string();

            let base = BaseCompletionRequest {
                model: model_name,
                prompt: prompt_for_cot.into(),

                ..Default::default()
            };

            let req = MyCompletionRequest {
                base,
            };


        }
        true
    }
}
