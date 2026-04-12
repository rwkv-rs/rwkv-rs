use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;

use super::{get_expected_context, get_judge_script};
use crate::cores::{
    datasets::{
        ALL_BENCHMARKS,
        Benchmark,
        BenchmarkInfo,
        BenchmarkName,
        CoTMode,
        Field,
        Record,
        SamplingConfig,
        coding::{extract_code, get_code_completion_with_cot_mode},
        utils::{
            collect_files_with_extension,
            hf::{download_hf_parquet_splits, viewer::get_split_row_count},
            parquet::{get_i64, get_string, get_string_list, read_parquet_items},
        },
    },
    evaluators::coding::run_python_verdict_script,
};

#[distributed_slice(ALL_BENCHMARKS)]
static MBPP_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mbpp"),
    field: Field::Coding,
    display_name: "MBPP",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
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
    with_llm_judger: false,
    create: |dataset_root| Box::new(Mbpp::new(dataset_root)),
};

pub struct Mbpp {
    dataset_root: PathBuf,
    test: Vec<MbppItem>,
}

pub struct MbppItem {
    task_id: String,
    prompt: String,
    code: String,
    test_imports: Vec<String>,
    test_list: Vec<String>,
}

impl Mbpp {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Mbpp {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join("mbpp/sanitized/test"), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        let parse_item = |row: &Row| MbppItem {
            task_id: get_i64(row, "task_id").to_string(),
            prompt: get_string(row, "prompt"),
            code: get_string(row, "code"),
            test_imports: get_string_list(row, "test_imports"),
            test_list: get_string_list(row, "test_list"),
        };
        for path in parquet_paths {
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.len()
            != get_split_row_count("google-research-datasets/mbpp", "sanitized", "test").await
    }

    async fn download(&self) {
        let downloaded_path = download_hf_parquet_splits(
            &self.dataset_root,
            "mbpp",
            "google-research-datasets/mbpp",
            "sanitized",
            &["test"],
            2,
        )
        .await;
        println!("mbpp dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];
        get_expected_context(&item.prompt, &item.code, cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].code.clone()
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        _judger_model_name: Option<&str>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        sandbox_queue: &crate::cores::sandbox_queue::SandboxQueue,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_code_completion_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &MBPP_INFO.sampling_config,
            cot_mode,
            1024,
        )
        .await;
        let answer = extract_code(&generated.completion);
        let verdict = run_python_verdict_script(
            &get_judge_script(&answer, &item.test_imports, &item.test_list, 3),
            sandbox_queue,
        )
        .await
        .unwrap_or_else(|err| {
            panic!(
                "mbpp sandbox execution failed: {err}; task={}",
                item.task_id
            )
        });

        let ref_answer = self.get_ref_answer(index);
        Record {
            context: generated.context,
            answer,
            ref_answer,
            is_passed: verdict.passed,
            fail_reason: if verdict.passed {
                String::new()
            } else {
                verdict.fail_reason
            },
        }
    }
}

