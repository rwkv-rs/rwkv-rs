use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;

use crate::datasets::{
    ALL_BENCHMARKS,
    Benchmark,
    BenchmarkInfo,
    BenchmarkName,
    CoTMode,
    Field,
    Record,
    SamplingConfig,
    knowledge::{Example, get_expect_context, get_final_answer_with_cot_mode, get_ref_answer},
    utils::{
        collect_files_with_extension,
        hf::{
            downloader::{UrlDownloadFile, download_url_files},
            viewer::{get_parquet_files, get_split_row_count},
        },
        parquet::{get_i64, get_string, get_string_list, read_parquet_items},
    },
};

const DATASET_ID: &str = "TIGER-Lab/MMLU-Pro";
const LOCAL_ROOT_NAME: &str = "mmlu_pro";

#[distributed_slice(ALL_BENCHMARKS)]
static MMLU_PRO_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mmlu_pro"),
    field: Field::Knowledge,
    display_name: "MMLU-Pro",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.5,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 1.0,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0, 5],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(MmluPro::new(dataset_root)),
};

pub struct MmluPro {
    dataset_root: PathBuf,
    validation: Vec<MmluProItem>,
    test: Vec<MmluProItem>,
}

#[allow(dead_code)]
pub struct MmluProItem {
    question_id: i64,
    question: String,
    options: Vec<String>,
    answer: String,
    answer_index: i64,
    cot_content: String,
    category: String,
    src: String,
}

impl MmluPro {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            validation: Vec::new(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for MmluPro {
    fn load(&mut self) -> bool {
        self.validation.clear();
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join(LOCAL_ROOT_NAME), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        for path in parquet_paths {
            let split = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap()
                .to_string();

            let items = read_parquet_items(&path, |row: &Row| MmluProItem {
                question_id: get_i64(row, "question_id"),
                question: get_string(row, "question"),
                options: get_string_list(row, "options"),
                answer: get_string(row, "answer"),
                answer_index: get_i64(row, "answer_index"),
                cot_content: get_string(row, "cot_content"),
                category: get_string(row, "category"),
                src: get_string(row, "src"),
            });

            match split.as_str() {
                "validation" => self.validation.extend(items),
                "test" => self.test.extend(items),
                _ => {}
            }
        }

        self.validation.is_empty() || self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let (remote_validation_len, remote_test_len) = async {
            tokio::join!(
                get_split_row_count(DATASET_ID, "default", "validation"),
                get_split_row_count(DATASET_ID, "default", "test"),
            )
        }
        .await;

        self.validation.len() != remote_validation_len || self.test.len() != remote_test_len
    }

    async fn download(&self) {
        let parquet_files = get_parquet_files(DATASET_ID)
            .await
            .into_iter()
            .filter(|file| file.config == "default")
            .collect::<Vec<_>>();
        let downloaded_path = download_url_files(
            &self.dataset_root,
            LOCAL_ROOT_NAME,
            &parquet_files
                .into_iter()
                .map(|file| UrlDownloadFile {
                    relative_path: file.relative_path(),
                    url: file.url,
                })
                .collect::<Vec<_>>(),
            4,
        )
        .await;
        println!("mmlu_pro dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        let item = &self.test[index];
        let few_shot_examples = self
            .validation
            .iter()
            .filter(|example| example.category == item.category)
            .take(n_shot as usize)
            .map(|example| Example {
                question: example.question.clone(),
                choices: example.options.clone(),
                answer_index: u8::try_from(example.answer_index).unwrap(),
            })
            .collect::<Vec<_>>();

        get_expect_context(
            &item.category,
            &item.question,
            &item.options,
            cot_mode,
            &few_shot_examples,
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        get_ref_answer(u8::try_from(self.test[index].answer_index).unwrap())
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
        let answer_index = u8::try_from(item.answer_index).unwrap();
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &item.options,
            &expected_context,
            &MMLU_PRO_INFO.sampling_config,
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
