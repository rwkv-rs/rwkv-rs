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
    knowledge::{Example, get_expect_context, get_final_answer_with_cot_mode, get_ref_answer},
    utils::{
        collect_files_with_extension,
        hf::{download_hf_parquet_splits, viewer::get_split_row_count},
        parquet::{get_string, get_string_list, get_u8, read_parquet_items},
    },
};

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
    n_shots: &[0, 5],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(Mmlu::new(dataset_root)),
};

pub struct Mmlu {
    dataset_root: PathBuf,
    dev: Vec<MmluItem>,
    test: Vec<MmluItem>,
}

pub struct MmluItem {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: u8,
}

impl Mmlu {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            dev: Vec::new(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Mmlu {
    fn load(&mut self) -> bool {
        self.dev.clear();
        self.test.clear();

        let dev_paths =
            collect_files_with_extension(self.dataset_root.join("mmlu/all/dev"), "parquet");
        let test_paths =
            collect_files_with_extension(self.dataset_root.join("mmlu/all/test"), "parquet");
        if dev_paths.is_empty() || test_paths.is_empty() {
            return true;
        }

        let parse_item = |row: &Row| MmluItem {
            question: get_string(row, "question"),
            subject: get_string(row, "subject"),
            choices: get_string_list(row, "choices"),
            answer: get_u8(row, "answer"),
        };
        for path in dev_paths {
            self.dev.extend(read_parquet_items(path, parse_item));
        }
        for path in test_paths {
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.dev.is_empty() || self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let (remote_dev_len, remote_test_len) = async {
            tokio::join!(
                get_split_row_count("cais/mmlu", "all", "dev"),
                get_split_row_count("cais/mmlu", "all", "test"),
            )
        }
        .await;

        self.dev.len() != remote_dev_len || self.test.len() != remote_test_len
    }

    async fn download(&self) {
        let downloaded_path = download_hf_parquet_splits(
            &self.dataset_root,
            "mmlu",
            "cais/mmlu",
            "all",
            &["dev", "test"],
            8,
        )
        .await;
        println!("mmlu dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        let item = &self.test[index];
        let few_shot_examples = self
            .dev
            .iter()
            .filter(|example| example.subject == item.subject)
            .take(n_shot as usize)
            .map(|example| Example {
                question: example.question.clone(),
                choices: example.choices.clone(),
                answer_index: example.answer,
            })
            .collect::<Vec<_>>();

        get_expect_context(
            &item.subject,
            &item.question,
            &item.choices,
            cot_mode,
            &few_shot_examples,
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        get_ref_answer(self.test[index].answer)
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
            &MMLU_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let ref_answer = self.get_ref_answer(index);
        let is_passed = generated.answer_index == item.answer;

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

#[cfg(test)]
mod tests {
    use super::MMLU_INFO;
    crate::cores::datasets::benchmark_dataset_tests!(MMLU_INFO);
}
