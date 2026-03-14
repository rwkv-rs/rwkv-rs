use super::human_eval_common::{get_check_script, get_prompt};
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::evaluators::coding::{get_completion, run_python_verdict_script};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use regex::Regex;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static HUMAN_EVAL_CN_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("human_eval_cn"),
    field: Field::Coding,
    display_name: "HumanEval-CN",
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
    create: |dataset_root| Box::new(HumanEvalCn::new(dataset_root)),
};

pub struct HumanEvalCn {
    dataset_root: PathBuf,
    test: Vec<HumanEvalCnItem>,
}

pub struct HumanEvalCnItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    test: String,
}

#[derive(Debug, Deserialize)]
struct RawHumanEvalCnItem {
    task_id: String,
    prompt: String,
    canonical_solution: String,
    test: String,
    declaration: Option<String>,
    entry_point: Option<String>,
}

impl HumanEvalCn {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn extract_entry_point(raw: &RawHumanEvalCnItem) -> String {
    if let Some(entry_point) = raw
        .entry_point
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return entry_point.trim().to_string();
    }

    let def_re = Regex::new(r"def\s+(?P<name>[\w_]+)\s*\(").unwrap();
    for text in [raw.declaration.as_deref(), Some(raw.prompt.as_str())]
        .into_iter()
        .flatten()
    {
        if let Some(capture) = def_re.captures(text) {
            return capture["name"].to_string();
        }
    }

    panic!(
        "failed to extract human_eval_cn entry point from task={}",
        raw.task_id
    );
}

#[async_trait]
impl Benchmark for HumanEvalCn {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("human_eval_cn")
            .join("humaneval.jsonl");
        if !path.is_file() {
            return true;
        }

        self.test = read_jsonl_items::<RawHumanEvalCnItem, _>(path)
            .into_iter()
            .map(|item| {
                let entry_point = extract_entry_point(&item);
                HumanEvalCnItem {
                    task_id: item.task_id,
                    prompt: item.prompt,
                    entry_point,
                    canonical_solution: item.canonical_solution,
                    test: item.test,
                }
            })
            .collect();

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.is_empty()
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            "human_eval_cn",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("humaneval.jsonl"),
                url: "https://hf-mirror.com/datasets/zai-org/humaneval-x/resolve/main/data/python/data/humaneval.jsonl"
                    .to_string(),
            }],
            1,
        ));
        println!("human_eval_cn dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        if cot_mode != CoTMode::NoCoT {
            panic!("human_eval_cn only supports NoCoT, got {cot_mode:?}");
        }

        get_prompt(&self.test[index].prompt, cot_mode, false)
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
        let completion = get_completion(
            model_client,
            model_name,
            &expected_context,
            &HUMAN_EVAL_CN_INFO.sampling_config,
            vec![],
            1024,
        )
        .await;
        let program = format!("{}{}", item.prompt, completion);
        let verdict = run_python_verdict_script(&get_check_script(
            &program,
            &item.test,
            &item.entry_point,
            3,
        ))
        .await
        .unwrap_or_else(|err| {
            panic!(
                "human_eval_cn sandbox execution failed: {err}; task={}",
                item.task_id
            )
        });

        verdict.passed
    }
}
