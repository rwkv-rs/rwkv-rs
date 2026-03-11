use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::hf_downloader::download_hf_files;
use crate::datasets::hf_viewer::get_split_row_count;
use crate::datasets::parquet_utils::{get_string, get_string_list, get_u8, read_parquet_items};
use crate::datasets::{ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig, get_prompt_for_cot, get_completions_of_cot, get_prompt_for_final_answer};
use crate::datasets::knowledge::{get_expected_context, get_final_answer, get_ref_answer};
use crate::inferers::{CompletionRequest, CompletionResponse};
use crate::runtime::OpenAiClient;

#[distributed_slice(ALL_BENCHMARKS)]
static MMLU_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmlu"),
    field: Field::Knowledge,
    display_name: "MMLU",
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
        get_expected_context(
            &item.subject,
            &item.question,
            &item.choices,
            cot_mode
        )
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        get_ref_answer(&item.answer)
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &OpenAiClient,
        _judger_client: Option<&OpenAiClient>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool {
        let expected_context = self.get_expected_context(item, cot_mode);
        let is_passed = match cot_mode {
            CoTMode::CoT => {
                let prompt_for_cot = get_prompt_for_cot(&expected_context);

                let completions_of_cot = get_completions_of_cot(
                    model_client,
                    &model_name,
                    &prompt_for_cot,
                    &MMLU_INFO.sampling_config,
                ).await;

                let prompt_for_final_answer = get_prompt_for_final_answer(
                    &expected_context,
                    Some(&completions_of_cot),
                );

                get_final_answer(
                    model_client,
                    &model_name,
                    &item.choices,
                    &prompt_for_final_answer,
                    &MMLU_INFO.sampling_config,
                ).await == item.answer
            }

            CoTMode::FakeCoT | CoTMode::NoCoT => {
                let prompt_for_final_answer = get_prompt_for_final_answer(
                    &expected_context,
                    None,
                );

                get_final_answer(
                    model_client,
                    &model_name,
                    &item.choices,
                    &prompt_for_final_answer,
                    &MMLU_INFO.sampling_config,
                ).await == item.answer
            }
        };
        is_passed
    }
}
