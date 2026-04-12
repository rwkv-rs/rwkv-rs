use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;

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
    utils::hf::downloader::{UrlDownloadFile, download_url_files},
};

#[distributed_slice(ALL_BENCHMARKS)]
static ASDIV_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("asdiv"),
    field: Field::Maths,
    display_name: "ASDiv",
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
    create: |dataset_root| Box::new(Asdiv::new(dataset_root)),
};

pub struct Asdiv {
    dataset_root: PathBuf,
    test: Vec<AsdivItem>,
}

pub struct AsdivItem {
    question: String,
    answer: String,
    subject: String,
}

impl Asdiv {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn extract_xml_text(block: &str, tag: &str) -> Option<String> {
    let (_, tail) = block.split_once(&format!("<{tag}>"))?;
    let (value, _) = tail.split_once(&format!("</{tag}>"))?;
    Some(
        value
            .trim()
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&"),
    )
}

#[async_trait]
impl Benchmark for Asdiv {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self.dataset_root.join("asdiv").join("ASDiv.xml");
        if !path.is_file() {
            return true;
        }

        self.test = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("open xml {} failed: {err}", path.display()))
            .split("<Problem ")
            .skip(1)
            .filter_map(|block| {
                let block = block.split_once("</Problem>")?.0;
                let body = extract_xml_text(block, "Body").unwrap_or_default();
                let question = extract_xml_text(block, "Question").unwrap_or_default();
                let subject = extract_xml_text(block, "Solution-Type")
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "math".to_string());
                let answer = extract_xml_text(block, "Answer")
                    .unwrap_or_default()
                    .split('(')
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let answer = answer
                    .parse::<f64>()
                    .ok()
                    .map(|value| {
                        if value.fract() == 0.0 {
                            (value as i64).to_string()
                        } else {
                            value.to_string()
                        }
                    })
                    .unwrap_or(answer);

                Some((
                    format!("{} {}", body.trim(), question.trim())
                        .trim()
                        .to_string(),
                    answer,
                    subject,
                ))
            })
            .map(|(question, answer, subject)| AsdivItem {
                question,
                answer,
                subject,
            })
            .collect();

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "asdiv",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("ASDiv.xml"),
                url: "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml".to_string(),
            }],
            1,
        ).await;
        println!("asdiv dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];

        get_expect_context(&item.subject, &item.question, cot_mode)
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
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &ASDIV_INFO.sampling_config,
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

