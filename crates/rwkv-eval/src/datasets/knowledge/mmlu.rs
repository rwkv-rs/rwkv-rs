use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::hf_downloader::download_hf_files;
use crate::datasets::hf_viewer::get_split_row_count;
use crate::datasets::parquet_utils::{get_string, get_string_list, get_u8, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
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
            CoTMode::NoCoT => "Assistant: Therefore, the answer is<|logprobs_of_choices|>",
            CoTMode::FakeCoT => concat!(
                "Assistant: <think>\n</think>\n",
                "Therefore, the answer is<|logprobs_of_choices|>",
            ),
            CoTMode::CoT => concat!(
                "Assistant: <think><|completions_of_cot|></think>\n",
                "Therefore, the answer is<|logprobs_of_choices|>"
            ),
        };

        format!("User: {user_part}\n\nAssistant: {assistant_part}")
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        let answer = item.answer;
        char::from(b'A' + answer).to_string()
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &OpenAiClient,
        judger_client: Option<&OpenAiClient>,
        cot_mode: CoTMode,
        fim_mode: bool,
        item: &Self::Item,
    ) -> bool {
        let expected_context = self.get_expected_context(item, cot_mode);
        let is_passed = match cot_mode {
            CoTMode::CoT => {
                let prompt_for_cot = expected_context
                    .split_once("<|completions_of_cot|>")
                    .unwrap()
                    .0
                    .to_string();

                let req = CompletionRequest::new(
                    model_name.clone(),
                    prompt_for_cot.into(),
                    vec!["</think>".to_string()],
                    4096,
                    &MMLU_INFO.sampling_config,
                    None,
                    None,
                );

                let resp: CompletionResponse = model_client.completions()
                    .create_byot(&req).await.unwrap();

                let completions_of_cot = &resp.choices[0].text;

                let prompt_for_final_answer = expected_context
                    .replace("<|completions_of_cot|>", completions_of_cot)
                    .split_once("<|logprobs_of_choices|>")
                    .unwrap()
                    .0
                    .to_string();

                let choice_token_texts = (0..item.choices.len())
                    .map(|i| format!(" {}", char::from(b'A' + i as u8)))
                    .collect::<Vec<_>>();

                let req = CompletionRequest::new(
                    model_name,
                    prompt_for_final_answer.into(),
                    vec![],
                    1,
                    &MMLU_INFO.sampling_config,
                    Some(1),
                    Some(choice_token_texts.clone()),
                );

                let resp: CompletionResponse = model_client.completions()
                    .create_byot(&req).await.unwrap();

                let choice_logprobs = resp.choices[0]
                    .logprobs
                    .as_ref()
                    .and_then(|logprobs| logprobs.top_logprobs.first())
                    .unwrap();

                let final_answer_id = choice_token_texts
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| {
                        choice_logprobs
                            .get(left.as_str())
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY)
                            .total_cmp(
                                &choice_logprobs
                                    .get(right.as_str())
                                    .copied()
                                    .unwrap_or(f32::NEG_INFINITY),
                            )
                    })
                    .map(|(idx, _)| idx)
                    .unwrap() as u8;

                final_answer_id == item.answer
            }

            CoTMode::FakeCoT => {
                let prompt_for_final_answer = expected_context
                    .split_once("<|logprobs_of_choices|>")
                    .unwrap()
                    .0
                    .to_string();

                let choice_token_texts = (0..item.choices.len())
                    .map(|i| format!(" {}", char::from(b'A' + i as u8)))
                    .collect::<Vec<_>>();

                let req = CompletionRequest::new(
                    model_name,
                    prompt_for_final_answer.into(),
                    vec![],
                    1,
                    &MMLU_INFO.sampling_config,
                    Some(1),
                    Some(choice_token_texts.clone()),
                );

                let resp: CompletionResponse = model_client.completions()
                    .create_byot(&req).await.unwrap();

                let choice_logprobs = resp.choices[0]
                    .logprobs
                    .as_ref()
                    .and_then(|logprobs| logprobs.top_logprobs.first())
                    .unwrap();

                let final_answer_id = choice_token_texts
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| {
                        choice_logprobs
                            .get(left.as_str())
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY)
                            .total_cmp(
                                &choice_logprobs
                                    .get(right.as_str())
                                    .copied()
                                    .unwrap_or(f32::NEG_INFINITY),
                            )
                    })
                    .map(|(idx, _)| idx)
                    .unwrap() as u8;

                final_answer_id == item.answer
            }

            CoTMode::NoCoT => {
                let prompt_for_final_answer = expected_context
                    .split_once("<|logprobs_of_choices|>")
                    .unwrap()
                    .0
                    .to_string();

                let choice_token_texts = (0..item.choices.len())
                    .map(|i| format!(" {}", char::from(b'A' + i as u8)))
                    .collect::<Vec<_>>();

                let req = CompletionRequest::new(
                    model_name,
                    prompt_for_final_answer.into(),
                    vec![],
                    1,
                    &MMLU_INFO.sampling_config,
                    Some(1),
                    Some(choice_token_texts.clone()),
                );

                let resp: CompletionResponse = model_client.completions()
                    .create_byot(&req).await.unwrap();

                let choice_logprobs = resp.choices[0]
                    .logprobs
                    .as_ref()
                    .and_then(|logprobs| logprobs.top_logprobs.first())
                    .unwrap();

                let final_answer_id = choice_token_texts
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| {
                        choice_logprobs
                            .get(left.as_str())
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY)
                            .total_cmp(
                                &choice_logprobs
                                    .get(right.as_str())
                                    .copied()
                                    .unwrap_or(f32::NEG_INFINITY),
                            )
                    })
                    .map(|(idx, _)| idx)
                    .unwrap() as u8;

                final_answer_id == item.answer
            }
        };
        is_passed
    }
}
