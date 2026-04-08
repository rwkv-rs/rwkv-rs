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
    maths::{get_expect_context, get_final_answer_with_cot_mode, judge_with_retry},
    utils::{
        collect_files_with_extension,
        hf::download_hf_parquet_splits,
        parquet::{get_string, read_parquet_items},
    },
};

#[distributed_slice(ALL_BENCHMARKS)]
static SIMPLEQA_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("simpleqa"),
    field: Field::Maths,
    display_name: "SimpleQA",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 500,
        top_p: 0.4,
        presence_penalty: 0.5,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[4.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(Simpleqa::new(dataset_root)),
};

pub struct Simpleqa {
    dataset_root: PathBuf,
    test: Vec<SimpleqaItem>,
}

pub struct SimpleqaItem {
    question: String,
    answer: String,
    subject: String,
}

impl Simpleqa {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Simpleqa {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths = collect_files_with_extension(
            self.dataset_root.join("simpleqa/default/train"),
            "parquet",
        );
        if parquet_paths.is_empty() {
            return true;
        }

        let parse_item = |row: &Row| SimpleqaItem {
            question: get_string(row, "problem"),
            answer: get_string(row, "answer"),
            subject: "qa".to_string(),
        };
        for path in parquet_paths {
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_hf_parquet_splits(
            &self.dataset_root,
            "simpleqa",
            "codelion/SimpleQA-Verified",
            "default",
            &["train"],
            2,
        )
        .await;
        println!("simpleqa dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.subject, &item.question, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].answer.clone()
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        judger_model_name: Option<&str>,
        judger_client: Option<&Client<OpenAIConfig>>,
        _sandbox_queue: &crate::cores::sandbox_queue::SandboxQueue,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &SIMPLEQA_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let judger_client = judger_client
            .unwrap_or_else(|| panic!("benchmark requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("benchmark requires judger_model_name but got None"));

        let ref_answer = self.get_ref_answer(index);
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            &expected_context,
            &ref_answer,
            &generated.answer,
        )
        .await;

        Record {
            context: generated.context,
            answer: generated.answer,
            ref_answer,
            is_passed: outcome.is_passed,
            fail_reason: outcome.fail_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SIMPLEQA_INFO;
    crate::cores::datasets::benchmark_dataset_tests!(SIMPLEQA_INFO);
}
