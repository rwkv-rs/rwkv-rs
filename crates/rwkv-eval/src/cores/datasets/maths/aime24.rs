use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
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
    maths::{get_expect_context, get_final_answer_with_cot_mode, judge_with_retry},
    utils::{
        hf::downloader::download_hf_files,
        parquet::{get_i64, get_string, read_parquet_items},
    },
};

#[distributed_slice(ALL_BENCHMARKS)]
static AIME24_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("aime24"),
    field: Field::Maths,
    display_name: "AIME24",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.55,
        top_k: 66,
        top_p: 0.79,
        presence_penalty: 0.14,
        repetition_penalty: 0.01,
        penalty_decay: 0.997,
    },
    n_shots: &[0],
    avg_ks: &[64.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(Aime24::new(dataset_root)),
};

pub struct Aime24 {
    dataset_root: PathBuf,
    test: Vec<Aime24Item>,
}

#[derive(Clone, Deserialize)]
pub struct Aime24Item {
    id: i64,
    problem: String,
    solution: String,
    url: String,
}

fn extract_boxed_answer(solution: &str) -> Option<String> {
    let start = solution.rfind(r"\boxed{")? + r"\boxed{".len();
    let rest = &solution[start..];
    let end = rest.find('}')?;
    Some(rest[..end].trim().to_string())
}

impl Aime24 {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Aime24 {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("aime24")
            .join("test-00000-of-00001.parquet");
        if !path.is_file() {
            return true;
        }

        let parse_item = |row: &Row| Aime24Item {
            id: get_i64(row, "id"),
            problem: get_string(row, "problem"),
            solution: get_string(row, "solution"),
            url: get_string(row, "url"),
        };
        self.test = read_parquet_items(path, parse_item);

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_hf_files(
            &self.dataset_root,
            "aime24",
            "datasets/math-ai/aime24",
            &["test-00000-of-00001.parquet"],
            1,
            "main",
        )
        .await;
        println!("aime24 dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.problem, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        extract_boxed_answer(&self.test[index].solution).unwrap_or_else(|| {
            panic!(
                "aime24 item {} missing \\boxed{{}} answer",
                self.test[index].id
            )
        })
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
            &AIME24_INFO.sampling_config,
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
    use super::{AIME24_INFO, extract_boxed_answer};

    #[test]
    fn extract_boxed_answer_works() {
        let solution = r"Some derivation ... \boxed{073}.";
        assert_eq!(extract_boxed_answer(solution).as_deref(), Some("073"));
        assert_eq!(extract_boxed_answer("no boxed answer here"), None);
    }

    crate::cores::datasets::benchmark_dataset_tests!(AIME24_INFO);
}
