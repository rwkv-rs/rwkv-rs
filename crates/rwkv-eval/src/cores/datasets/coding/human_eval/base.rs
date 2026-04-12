use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;

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
            hf::downloader::{UrlDownloadFile, download_url_files},
            jsonl::read_gzip_jsonl_items,
        },
    },
    evaluators::coding::run_python_verdict_script,
};

#[distributed_slice(ALL_BENCHMARKS)]
static HUMAN_EVAL_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("human_eval"),
    field: Field::Coding,
    display_name: "HumanEval",
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
    create: |dataset_root| Box::new(HumanEval::new(dataset_root)),
};

pub struct HumanEval {
    dataset_root: PathBuf,
    test: Vec<HumanEvalItem>,
}

#[derive(Debug, Deserialize)]
pub struct HumanEvalItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    test: String,
}

impl HumanEval {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for HumanEval {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("human_eval")
            .join("HumanEval.jsonl.gz");
        if !path.is_file() {
            return true;
        }

        self.test = read_gzip_jsonl_items(path);

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "human_eval",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("HumanEval.jsonl.gz"),
                url: "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
                    .to_string(),
            }],
            1,
        )
        .await;
        println!("human_eval dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        if cot_mode != CoTMode::NoCoT {
            panic!("human_eval only supports NoCoT, got {cot_mode:?}");
        }

        let item = &self.test[index];
        get_expected_context(&item.prompt, Some(item.prompt.as_str()), cot_mode)
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
    ) -> Record {
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_code_completion_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &HUMAN_EVAL_INFO.sampling_config,
            cot_mode,
            1024,
        )
        .await;
        let completion = extract_code(&generated.completion);
        let answer = format!("{}{}", item.prompt, completion);
        let verdict =
            run_python_verdict_script(&get_judge_script(&answer, &item.test, &item.entry_point, 3))
                .await
                .unwrap_or_else(|err| {
                    panic!(
                        "human_eval sandbox execution failed: {err}; task={}",
                        item.task_id
                    )
                });

        Record {
            context: generated.context,
            answer,
            ref_answer: self.get_ref_answer(index),
            is_passed: verdict.passed,
            fail_reason: if verdict.passed {
                String::new()
            } else {
                verdict.fail_reason
            },
        }
    }
}

