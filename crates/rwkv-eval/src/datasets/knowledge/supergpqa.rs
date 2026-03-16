use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::datasets::knowledge::gpqa_common::join_subject_parts;
use crate::datasets::knowledge::{
    answer_index_from_letter, get_expect_context, get_final_answer_with_cot_mode, get_ref_answer,
};
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};

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
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(SuperGpqa::new(dataset_root)),
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

#[async_trait]
impl Benchmark for SuperGpqa {
    fn load(&mut self) -> bool {
        let jsonl_path = self.dataset_root.join(LOCAL_ROOT_NAME).join(FILE_NAME);
        if !jsonl_path.is_file() {
            return true;
        }

        self.test = read_jsonl_items(jsonl_path);

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            LOCAL_ROOT_NAME,
            &[UrlDownloadFile {
                relative_path: PathBuf::from(FILE_NAME),
                url: FILE_URL.to_string(),
            }],
            1,
        )
        .await;
        println!("supergpqa dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];
        let subject = join_subject_parts(&[&item.discipline, &item.field, &item.subfield]);
        get_expect_context(&subject, &item.question, &item.options, cot_mode, &[])
    }

    fn get_ref_answer(&self, index: usize) -> String {
        get_ref_answer(answer_index_from_letter(&self.test[index].answer_letter))
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
        let answer_index = answer_index_from_letter(&item.answer_letter);
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &item.options,
            &expected_context,
            &SUPERGPQA_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let ref_answer = self.get_ref_answer(index);
        let is_passed = generated.answer_index == answer_index;

        Record {
            context: generated.context,
            answer: generated.answer_text.clone(),
            ref_answer: ref_answer.clone(),
            is_passed,
            fail_reason: if is_passed {
                String::new()
            } else {
                format!(
                    "predicted {}, expected {}",
                    generated.answer_text, ref_answer
                )
            },
        }
    }
}
