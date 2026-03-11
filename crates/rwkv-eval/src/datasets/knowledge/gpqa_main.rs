use async_openai::Client;
use async_openai::config::OpenAIConfig;
use linkme::distributed_slice;
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::datasets::knowledge::gpqa_common::{
    download_gpqa_csv, gpqa_csv_path, join_subject_parts, ordered_gpqa_choices,
};
use crate::datasets::knowledge::{
    get_ref_answer, get_final_answer_with_cot_mode, get_expect_context,
};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::datasets::utils::csv::read_csv_items;


#[distributed_slice(ALL_BENCHMARKS)]
static GPQA_MAIN_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("gpqa_main"),
    field: Field::Knowledge,
    display_name: "GPQA Main",
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

pub struct GpqaMain {
    dataset_root: PathBuf,
    test: Vec<GpqaMainItem>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GpqaMainItem {
    #[serde(rename = "Question")]
    question: String,
    #[serde(rename = "Correct Answer")]
    correct_answer: String,
    #[serde(rename = "Incorrect Answer 1")]
    incorrect_answer_1: String,
    #[serde(rename = "Incorrect Answer 2")]
    incorrect_answer_2: String,
    #[serde(rename = "Incorrect Answer 3")]
    incorrect_answer_3: String,
    #[serde(rename = "Explanation")]
    explanation: String,
    #[serde(rename = "Subdomain")]
    subdomain: String,
    #[serde(rename = "High-level domain")]
    high_level_domain: String,
    #[serde(rename = "Record ID")]
    record_id: String,
    #[serde(rename = "Question Writer")]
    question_writer: Option<String>,
}

impl GpqaMain {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

impl Benchmark for GpqaMain {
    type Item = GpqaMainItem;

    fn load(&mut self) {
        self.test = read_csv_items(gpqa_csv_path(&self.dataset_root, "gpqa_main.csv"));
    }

    fn check(&self) -> bool {
        !gpqa_csv_path(&self.dataset_root, "gpqa_main.csv").exists()
    }

    fn download(&self) {
        let downloaded_path = download_gpqa_csv(&self.dataset_root, "gpqa_main.csv");
        println!("gpqa_main dataset: {}", downloaded_path.display());
    }

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String {
        let subject = join_subject_parts(&[&item.high_level_domain, &item.subdomain]);
        let (choices, _) = ordered_gpqa_choices(
            &item.record_id,
            &item.question,
            &item.correct_answer,
            [
                &item.incorrect_answer_1,
                &item.incorrect_answer_2,
                &item.incorrect_answer_3,
            ],
        );

        get_expect_context(&subject, &item.question, &choices, cot_mode)
    }

    fn get_ref_answer(&self, item: &Self::Item) -> String {
        let (_, answer_index) = ordered_gpqa_choices(
            &item.record_id,
            &item.question,
            &item.correct_answer,
            [
                &item.incorrect_answer_1,
                &item.incorrect_answer_2,
                &item.incorrect_answer_3,
            ],
        );
        get_ref_answer(answer_index)
    }

    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &Client<OpenAIConfig>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool {
        let (choices, answer_index) = ordered_gpqa_choices(
            &item.record_id,
            &item.question,
            &item.correct_answer,
            [
                &item.incorrect_answer_1,
                &item.incorrect_answer_2,
                &item.incorrect_answer_3,
            ],
        );
        let subject = join_subject_parts(&[&item.high_level_domain, &item.subdomain]);
        let expected_context =
            get_expect_context(&subject, &item.question, &choices, cot_mode);

        get_final_answer_with_cot_mode(
            model_client,
            &model_name,
            &choices,
            &expected_context,
            &GPQA_MAIN_INFO.sampling_config,
            cot_mode,
        ).await == answer_index
    }
}
