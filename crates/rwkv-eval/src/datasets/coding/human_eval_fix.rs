use super::human_eval_common::{get_expected_context, get_judge_script};
use crate::datasets::coding::{extract_code, get_code_completion_with_cot_mode};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::download_hf_parquet_splits;
use crate::datasets::utils::hf::viewer::get_split_row_count;
use crate::datasets::utils::parquet::{get_string, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::evaluators::coding::run_python_verdict_script;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static HUMAN_EVAL_FIX_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("human_eval_fix"),
    field: Field::Coding,
    display_name: "HumanEvalFix",
    cot_mode: &[CoTMode::NoCoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 500,
        top_p: 0.4,
        presence_penalty: 0.5,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(HumanEvalFix::new(dataset_root)),
};

pub struct HumanEvalFix {
    dataset_root: PathBuf,
    test: Vec<HumanEvalFixItem>,
}

pub struct HumanEvalFixItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    test: String,
}

impl HumanEvalFix {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for HumanEvalFix {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths = collect_files_with_extension(
            self.dataset_root.join("human_eval_fix/python/test"),
            "parquet",
        );
        if parquet_paths.is_empty() {
            return true;
        }

        let parse_item = |row: &Row| {
            let prompt = get_string(row, "prompt");
            let buggy_solution = get_string(row, "buggy_solution");
            let entry_point = get_string(row, "entry_point");
            let mut parts = Vec::new();

            if !prompt.trim_end().is_empty() {
                parts.push(prompt.trim_end().to_string());
            }
            if !buggy_solution.trim_end().is_empty() {
                parts.push("# Buggy implementation:".to_string());
                parts.push(buggy_solution.trim_end().to_string());
            }
            if !entry_point.trim().is_empty() {
                parts.push(format!(
                    "# Fix the function `{}` so it passes all tests.",
                    entry_point.trim()
                ));
            }

            HumanEvalFixItem {
                task_id: get_string(row, "task_id"),
                prompt: parts.join("\n"),
                entry_point,
                canonical_solution: get_string(row, "canonical_solution"),
                test: get_string(row, "test"),
            }
        };
        for path in parquet_paths {
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        let runtime = Runtime::new().unwrap();
        self.test.len()
            != runtime.block_on(get_split_row_count(
                "bigcode/humanevalpack",
                "python",
                "test",
            ))
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_hf_parquet_splits(
            &self.dataset_root,
            "human_eval_fix",
            "bigcode/humanevalpack",
            "python",
            &["test"],
            2,
        ));
        println!("human_eval_fix dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        if cot_mode != CoTMode::NoCoT {
            panic!("human_eval_fix only supports NoCoT, got {cot_mode:?}");
        }

        get_expected_context(&self.test[index].prompt, None, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].canonical_solution.clone()
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
    ) -> bool {
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let completion = get_code_completion_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &HUMAN_EVAL_FIX_INFO.sampling_config,
            cot_mode,
            1024,
        )
        .await;
        let completion = extract_code(&completion);
        let verdict = run_python_verdict_script(&get_judge_script(
            &completion,
            &item.test,
            &item.entry_point,
            3,
        ))
        .await
        .unwrap_or_else(|err| {
            panic!(
                "human_eval_fix sandbox execution failed: {err}; task={}",
                item.task_id
            )
        });

        verdict.passed
    }
}
