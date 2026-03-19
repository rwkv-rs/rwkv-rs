use super::browsecomp_common::{
    BrowseCompLocale, browsecomp_sample_limit, build_browsecomp_expected_context, decrypt_xor_base64,
    generate_browsecomp_answer, judge_with_retry,
};
use crate::datasets::utils::csv::read_csv_items;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use std::path::{Path, PathBuf};

const BROWSECOMP_EXPECTED_LEN: usize = 1266;

#[distributed_slice(ALL_BENCHMARKS)]
static BROWSECOMP_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("browsecomp"),
    field: Field::FunctionCalling,
    display_name: "BrowseComp",
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
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(BrowseComp::new(dataset_root)),
};

pub struct BrowseComp {
    dataset_root: PathBuf,
    test: Vec<BrowseCompItem>,
}

pub struct BrowseCompItem {
    question: String,
    answer: String,
}

#[derive(Debug, Deserialize)]
struct RawBrowseCompRow {
    problem: String,
    answer: String,
    canary: String,
}

impl BrowseComp {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }

    fn load_items(&self) -> Result<Vec<BrowseCompItem>, String> {
        let path = self
            .dataset_root
            .join("browsecomp")
            .join("browse_comp_test_set.csv");
        if !path.is_file() {
            return Err(format!("missing browsecomp csv: {}", path.display()));
        }

        read_csv_items::<RawBrowseCompRow, _>(&path)
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let question = decrypt_xor_base64(&row.problem, &row.canary)
                    .map_err(|err| format!("row {index} problem decrypt failed: {err}"))?;
                let answer = decrypt_xor_base64(&row.answer, &row.canary)
                    .map_err(|err| format!("row {index} answer decrypt failed: {err}"))?;
                Ok(BrowseCompItem { question, answer })
            })
            .take(browsecomp_sample_limit().unwrap_or(usize::MAX))
            .collect()
    }
}

fn build_system_prompt() -> &'static str {
    "You are a browsing benchmark assistant. Think through the question carefully and then answer directly."
}

fn build_user_prompt(question: &str) -> String {
    format!(
        concat!(
            "Answer the following browsing-intensive question using your own knowledge.\n",
            "Do not refuse by asking the user to search the web themselves.\n",
            "If you are uncertain, still provide your best concrete answer.\n\n",
            "Question:\n{question}\n\n",
            "Return your final answer in this format:\n",
            "Explanation: <brief explanation>\n",
            "Exact Answer: <succinct final answer>\n",
            "Confidence: <0% to 100%>"
        ),
        question = question
    )
}

#[async_trait]
impl Benchmark for BrowseComp {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("browsecomp")
            .join("browse_comp_test_set.csv");
        if !path.is_file() {
            return true;
        }

        self.test = self
            .load_items()
            .unwrap_or_else(|err| panic!("failed to load browsecomp: {err}"));
        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let size_invalid = if let Some(limit) = browsecomp_sample_limit() {
            self.test.len() != limit.min(BROWSECOMP_EXPECTED_LEN)
        } else {
            self.test.len() != BROWSECOMP_EXPECTED_LEN
        };

        size_invalid
            || self
                .test
                .iter()
                .any(|item| item.question.trim().is_empty() || item.answer.trim().is_empty())
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "browsecomp",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("browse_comp_test_set.csv"),
                url: "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv".to_string(),
            }],
            1,
        )
        .await;
        println!("browsecomp dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(cot_mode, CoTMode::CoT, "browsecomp only supports CoT");
        assert_eq!(n_shot, 0, "browsecomp only supports 0-shot");

        build_browsecomp_expected_context(
            build_system_prompt(),
                &build_user_prompt(&self.test[index].question),
        )
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
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let ref_answer = self.get_ref_answer(index);
        let (context, answer) = generate_browsecomp_answer(
            model_client,
            model_name,
            &expected_context,
            &BROWSECOMP_INFO.sampling_config,
        )
        .await;

        if answer.trim().is_empty() {
            return Record {
                context,
                answer,
                ref_answer,
                is_passed: false,
                fail_reason: "model returned empty response".to_string(),
            };
        }

        let judger_client = judger_client
            .unwrap_or_else(|| panic!("browsecomp requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("browsecomp requires judger_model_name but got None"));
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            BrowseCompLocale::En,
            &item.question,
            &answer,
            &item.answer,
        )
        .await;

        Record {
            context,
            answer,
            ref_answer,
            is_passed: outcome.is_passed,
            fail_reason: if outcome.is_passed {
                String::new()
            } else if outcome.reason.is_empty() {
                "judger marked answer incorrect".to_string()
            } else {
                outcome.reason
            },
        }
    }
}
