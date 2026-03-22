use crate::datasets::maths::{get_expect_context, get_final_answer_with_cot_mode, judge_with_retry};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::download_hf_parquet_splits;
use crate::datasets::utils::parquet::{get_string, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::{Field as ParquetField, Row};
use std::path::{Path, PathBuf};

#[distributed_slice(ALL_BENCHMARKS)]
static ANSWER_JUDGE_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("answer_judge"),
    field: Field::Maths,
    display_name: "Answer-Judge",
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
    create: |dataset_root| Box::new(AnswerJudge::new(dataset_root)),
};

pub struct AnswerJudge {
    dataset_root: PathBuf,
    test: Vec<AnswerJudgeItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn downloads_and_reads_dataset() {
        crate::datasets::assert_benchmark_download_load_and_read(&ANSWER_JUDGE_INFO).await;
    }
}

pub struct AnswerJudgeItem {
    question: String,
    answer: String,
    subject: String,
}

impl AnswerJudge {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn score_from_field(field: &ParquetField) -> Option<f64> {
    match field {
        ParquetField::Str(value) => value.parse().ok(),
        ParquetField::Byte(value) => Some(f64::from(*value)),
        ParquetField::Short(value) => Some(f64::from(*value)),
        ParquetField::Int(value) => Some(f64::from(*value)),
        ParquetField::Long(value) => Some(*value as f64),
        ParquetField::UByte(value) => Some(f64::from(*value)),
        ParquetField::UShort(value) => Some(f64::from(*value)),
        ParquetField::UInt(value) => Some(f64::from(*value)),
        ParquetField::ULong(value) => Some(*value as f64),
        ParquetField::Float(value) => Some(f64::from(*value)),
        ParquetField::Double(value) => Some(*value),
        _ => None,
    }
}

fn mean_annotation_score(row: &Row) -> Option<f64> {
    let annotations = row
        .get_column_iter()
        .find(|(name, _)| name.as_str() == "annotations")
        .map(|(_, field)| field)?;
    let ParquetField::ListInternal(annotations) = annotations else {
        return None;
    };

    let scores = annotations
        .elements()
        .iter()
        .filter_map(|annotation| {
            let ParquetField::Group(annotation) = annotation else {
                return None;
            };
            annotation
                .get_column_iter()
                .find(|(name, _)| name.as_str() == "score")
                .and_then(|(_, field)| score_from_field(field))
        })
        .collect::<Vec<_>>();

    (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64)
}

#[async_trait]
impl Benchmark for AnswerJudge {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths = collect_files_with_extension(
            self.dataset_root.join("answer_judge/default/train"),
            "parquet",
        );
        if parquet_paths.is_empty() {
            return true;
        }

        let parse_item = |row: &Row| {
            let item_name = get_string(row, "item_name");
            let question = get_string(row, "question");
            let reference_answer = get_string(row, "gt_answer");
            let predicted_answer = get_string(row, "gen_answer");
            let score = mean_annotation_score(row).unwrap_or_else(|| {
                panic!("judges-verdict missing valid annotations for item {item_name}")
            });
            let expected = if score > 0.5 {
                "Judgement: Yes"
            } else {
                "Judgement: No"
            };

            AnswerJudgeItem {
                question: format!(
                    concat!(
                        "Question: {question}\n",
                        "Reference Answer: {reference_answer}\n",
                        "Student Answer: {predicted_answer}\n",
                        "Judge whether the student answer is correct.",
                    ),
                    question = question,
                    reference_answer = reference_answer,
                    predicted_answer = predicted_answer,
                ),
                answer: expected.to_string(),
                subject: "judgement".to_string(),
            }
        };
        for path in parquet_paths {
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.is_empty()
    }

    async fn download(&self) {
        let downloaded_path = download_hf_parquet_splits(
            &self.dataset_root,
            "answer_judge",
            "nvidia/judges-verdict",
            "default",
            &["train"],
            2,
        )
        .await;
        println!("answer_judge dataset: {}", downloaded_path.display());
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
            &ANSWER_JUDGE_INFO.sampling_config,
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
