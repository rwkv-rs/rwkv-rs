use async_openai::Client;
use async_openai::config::OpenAIConfig;
use linkme::distributed_slice;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

use crate::datasets::knowledge::{
    get_expected_context, get_ref_answer_from_letter, join_subject_parts,
    judge_multiple_choice_by_letter,
};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;

const LOCAL_ROOT_NAME: &str = "supergpqa";
const FILE_NAME: &str = "SuperGPQA-all.jsonl";
const FILE_URL: &str =
    "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl";

#[distributed_slice(ALL_BENCHMARKS)]
static SUPERGPQA_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("supergpqa"),
    field: Field::Knowledge,
    display_name: "SuperGPQA",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.5,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 1.0,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    avg_ks: &[1],
    pass_ks: &[1],
    with_llm_judger: false,
};

pub struct SuperGpqa {
    dataset_root: PathBuf,
    test: Vec<SuperGpqaItem>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SuperGpqaItem {
    uuid: String,
    question: String,
    options: Vec<String>,
    answer: String,
    answer_letter: String,
    discipline: String,
    field: String,
    subfield: String,
    difficulty: String,
    is_calculation: bool,
}

impl SuperGpqa {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

impl Benchmark for SuperGpqa {
    type Item = SuperGpqaItem;

    fn load(&mut self) {
        self.test = read_jsonl_items(self.dataset_root.join(LOCAL_ROOT_NAME).join(FILE_NAME));
    }

    fn check(&self) -> bool {
        !self.dataset_root.join(LOCAL_ROOT_NAME).join(FILE_NAME).exists()
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            LOCAL_ROOT_NAME,
            &[UrlDownloadFile {
                relative_path: PathBuf::from(FILE_NAME),
                url: FILE_URL.to_string(),
            }],
            1,
        ));
        println!("supergpqa dataset: {}", downloaded_path.display());
    }

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        let subject = join_subject_parts(&[&item.discipline, &item.field, &item.subfield]);
        get_expected_context(&subject, &item.question, &item.options, cot_mode)
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        get_ref_answer_from_letter(&item.answer_letter)
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &Client<OpenAIConfig>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool {
        let expected_context = self.get_expected_context(item, cot_mode);

        judge_multiple_choice_by_letter(
            model_client,
            &model_name,
            &item.options,
            &expected_context,
            &SUPERGPQA_INFO.sampling_config,
            cot_mode,
            &item.answer_letter,
        )
        .await
    }
}
