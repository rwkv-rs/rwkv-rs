use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use serde::Deserialize;
use sonic_rs::Value;

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
        hf::{download_hf_parquet_splits, downloader::download_hf_files},
        jsonl::read_jsonl_items,
        parquet::{get_i64, get_string, read_parquet_items},
    },
};

#[distributed_slice(ALL_BENCHMARKS)]
static COMP_MATH_24_25_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("comp_math_24_25"),
    field: Field::Maths,
    display_name: "Comp-Math-24-25",
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
    avg_ks: &[16.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(CompMath2425::new(dataset_root)),
};

pub struct CompMath2425 {
    dataset_root: PathBuf,
    test: Vec<CompMath2425Item>,
}

pub enum CompMath2425Item {
    Aime24(Aime24Item),
    Aime25(Aime25Item),
    Hmmt(HmmtFeb2025Item),
}

#[derive(Clone, Deserialize)]
pub struct Aime24Item {
    id: i64,
    problem: String,
    solution: String,
    url: String,
}

#[derive(Clone, Deserialize)]
pub struct Aime25Item {
    id: String,
    problem: String,
    answer: Value,
}

pub struct HmmtFeb2025Item {
    id: String,
    problem: String,
    answer: String,
    problem_type: Option<Vec<String>>,
}

fn extract_boxed_answer(solution: &str) -> Option<String> {
    let start = solution.rfind(r"\boxed{")? + r"\boxed{".len();
    let rest = &solution[start..];
    let end = rest.find('}')?;
    Some(rest[..end].trim().to_string())
}

impl CompMath2425 {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for CompMath2425 {
    fn load(&mut self) -> bool {
        self.test.clear();

        let aime24_path = self
            .dataset_root
            .join("comp_math_24_25")
            .join("aime24")
            .join("test-00000-of-00001.parquet");
        let aime25_path = self
            .dataset_root
            .join("comp_math_24_25")
            .join("aime25")
            .join("test.jsonl");
        let hmmt_dir = self
            .dataset_root
            .join("comp_math_24_25")
            .join("default")
            .join("train");
        let hmmt_paths = collect_files_with_extension(hmmt_dir, "parquet");
        if !aime24_path.is_file() || !aime25_path.is_file() || hmmt_paths.is_empty() {
            return true;
        }

        let parse_hmmt_item = |row: &Row| HmmtFeb2025Item {
            id: get_string(row, "id"),
            problem: get_string(row, "problem"),
            answer: get_string(row, "answer"),
            problem_type: row
                .get_column_iter()
                .find(|(name, _)| name.as_str() == "problem_type")
                .map(|_| {
                    crate::cores::datasets::utils::parquet::get_optional_string_list(
                        row,
                        "problem_type",
                    )
                })
                .unwrap_or(None),
        };
        self.test.extend(
            read_parquet_items::<Aime24Item, _, _>(aime24_path, |row| Aime24Item {
                id: get_i64(row, "id"),
                problem: get_string(row, "problem"),
                solution: get_string(row, "solution"),
                url: get_string(row, "url"),
            })
            .into_iter()
            .map(CompMath2425Item::Aime24),
        );
        self.test.extend(
            read_jsonl_items::<Aime25Item, _>(aime25_path)
                .into_iter()
                .map(CompMath2425Item::Aime25),
        );
        for path in hmmt_paths {
            self.test.extend(
                read_parquet_items(path, parse_hmmt_item)
                    .into_iter()
                    .map(CompMath2425Item::Hmmt),
            );
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_hf_files(
            &self.dataset_root,
            "comp_math_24_25",
            "datasets/math-ai/aime24",
            &["aime24/test-00000-of-00001.parquet"],
            1,
            "main",
        )
        .await;
        download_hf_files(
            &self.dataset_root,
            "comp_math_24_25",
            "datasets/math-ai/aime25",
            &["aime25/test.jsonl"],
            1,
            "main",
        )
        .await;
        download_hf_parquet_splits(
            &self.dataset_root,
            "comp_math_24_25",
            "MathArena/hmmt_feb_2025",
            "default",
            &["train"],
            2,
        )
        .await;
        println!("comp_math_24_25 dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        let problem = match item {
            CompMath2425Item::Aime24(row) => &row.problem,
            CompMath2425Item::Aime25(row) => &row.problem,
            CompMath2425Item::Hmmt(row) => &row.problem,
        };
        get_expect_context(problem, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        match &self.test[index] {
            CompMath2425Item::Aime24(row) => extract_boxed_answer(&row.solution)
                .unwrap_or_else(|| panic!("aime24 item {} missing \\boxed{{}} answer", row.id)),
            CompMath2425Item::Aime25(row) => {
                crate::cores::datasets::maths::json_value_as_text(&row.answer)
                    .unwrap_or_else(|| panic!("aime25 item has unsupported answer: {}", row.id))
            }
            CompMath2425Item::Hmmt(row) => row.answer.clone(),
        }
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
            &COMP_MATH_24_25_INFO.sampling_config,
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
    use sonic_rs::json;

    use super::{COMP_MATH_24_25_INFO, extract_boxed_answer};

    #[test]
    fn extract_boxed_answer_works() {
        let solution = r"Intermediate steps \boxed{588}";
        assert_eq!(extract_boxed_answer(solution).as_deref(), Some("588"));
    }

    #[test]
    fn json_value_to_text_for_aime25_answer() {
        let answer = json!(16);
        assert_eq!(
            crate::cores::datasets::maths::json_value_as_text(&answer).as_deref(),
            Some("16")
        );
    }

    crate::cores::datasets::benchmark_dataset_tests!(COMP_MATH_24_25_INFO);
}
