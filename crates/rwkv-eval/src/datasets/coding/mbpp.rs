use super::mbpp_common::{get_assertion_script, get_prompt};
use crate::datasets::coding::extract_code;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::evaluators::coding::{get_completion, run_python_verdict_script};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static MBPP_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mbpp"),
    field: Field::Coding,
    display_name: "MBPP",
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

#[derive(Debug, Deserialize)]
struct RawMbppItem {
    task_id: serde_json::Value,
    prompt: String,
    code: String,
    #[serde(default)]
    test_imports: Vec<String>,
    #[serde(default)]
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

        let path = self.dataset_root.join("mbpp").join("sanitized-mbpp.json");
        if !path.is_file() {
            return true;
        }

        let file = File::open(path).unwrap();
        self.test = serde_json::from_reader::<_, Vec<RawMbppItem>>(file)
            .unwrap()
            .into_iter()
            .map(|item| MbppItem {
                task_id: item.task_id.to_string().trim_matches('"').to_string(),
                prompt: item.prompt,
                code: item.code,
                test_imports: item.test_imports,
                test_list: item.test_list,
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
            "mbpp",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("sanitized-mbpp.json"),
                url: "https://github.com/google-research/google-research/raw/master/mbpp/sanitized-mbpp.json"
                    .to_string(),
            }],
            1,
        ));
        println!("mbpp dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        if cot_mode != CoTMode::NoCoT {
            panic!("mbpp only supports NoCoT, got {cot_mode:?}");
        }

        let item = &self.test[index];
        get_prompt(&item.prompt, &item.code, cot_mode, true)
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
            &MBPP_INFO.sampling_config,
            vec!["```".to_string()],
            1024,
        )
        .await;
        let completion = extract_code(&completion);
        let verdict = run_python_verdict_script(&get_assertion_script(
            &completion,
            &item.test_imports,
            &item.test_list,
            3,
        ))
        .await
        .unwrap_or_else(|err| {
            panic!(
                "mbpp sandbox execution failed: {err}; task={}",
                item.task_id
            )
        });

        verdict.passed
    }
}
