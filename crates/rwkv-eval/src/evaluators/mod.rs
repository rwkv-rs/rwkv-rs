use async_trait::async_trait;

use crate::checkers::{Checker, LlmChecker};
use crate::datasets::{Benchmark, BenchmarkSplit};
use crate::error::EvalError;
use crate::runtime::{EvalRuntime, ModelRuntime};

#[derive(Clone, Debug)]
pub enum JudgementSource {
    ExactMatch,
    LlmJudger,
}

#[derive(Clone, Debug)]
pub struct EvaluationSampleResult {
    pub split: BenchmarkSplit,
    pub index: usize,
    pub context: String,
    pub completion_text: String,
    pub final_answer: String,
    pub ref_answer: String,
    pub is_pass: bool,
    pub judgement_source: JudgementSource,
    pub error_type: Option<String>,
}

#[async_trait]
pub trait Evaluator: Send + Sync {
    async fn evaluate(
        &self,
        runtime: &EvalRuntime,
        benchmark: &dyn Benchmark,
        model: &ModelRuntime,
        split: BenchmarkSplit,
        index: usize,
    ) -> Result<EvaluationSampleResult, EvalError>;
}

pub struct MmluEvaluator;

#[async_trait]
impl Evaluator for MmluEvaluator {
    async fn evaluate(
        &self,
        runtime: &EvalRuntime,
        benchmark: &dyn Benchmark,
        model: &ModelRuntime,
        split: BenchmarkSplit,
        index: usize,
    ) -> Result<EvaluationSampleResult, EvalError> {
        let context = benchmark.get_expected_context(split, index);
        let ref_answer = benchmark.get_ref_answer(split, index);
        let completion_text = runtime.complete_with_model(model, &context).await?;
        let final_answer = extract_multiple_choice_answer(&completion_text)
            .unwrap_or_else(|| completion_text.trim().to_string());

        let mut is_pass = exact_match(&final_answer, &ref_answer);
        let mut judgement_source = JudgementSource::ExactMatch;
        if !is_pass && benchmark.with_llm_judger() {
            is_pass = runtime
                .judge_answer(benchmark.name(), &context, &ref_answer, &final_answer)
                .await?;
            judgement_source = JudgementSource::LlmJudger;
        }

        let checker = LlmChecker;
        let mut result = EvaluationSampleResult {
            split,
            index,
            context,
            completion_text,
            final_answer,
            ref_answer,
            is_pass,
            judgement_source,
            error_type: None,
        };

        result.error_type = checker
            .classify_error(runtime, benchmark.name(), &result)
            .await?;

        Ok(result)
    }
}

pub fn exact_match(left: &str, right: &str) -> bool {
    normalize_answer(left) == normalize_answer(right)
}

fn normalize_answer(answer: &str) -> String {
    answer
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
        .to_ascii_uppercase()
}

pub fn extract_multiple_choice_answer(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    for marker in [
        "answer is",
        "Answer:",
        "Answer is",
        "Final answer:",
        "Therefore",
    ] {
        if let Some((_, tail)) = trimmed.rsplit_once(marker) {
            if let Some(choice) = extract_choice_from_fragment(tail) {
                return Some(choice);
            }
        }
    }

    extract_choice_from_fragment(trimmed)
}

fn extract_choice_from_fragment(fragment: &str) -> Option<String> {
    fragment
        .chars()
        .find(|ch| matches!(ch.to_ascii_uppercase(), 'A' | 'B' | 'C' | 'D'))
        .map(|ch| ch.to_ascii_uppercase().to_string())
}

#[cfg(test)]
mod tests {
    use super::{exact_match, extract_multiple_choice_answer};

    #[test]
    fn extract_multiple_choice_answer_prefers_final_marker() {
        assert_eq!(
            extract_multiple_choice_answer("Reasoning... Therefore, the answer is C."),
            Some("C".to_string())
        );
    }

    #[test]
    fn exact_match_ignores_spacing_and_case() {
        assert!(exact_match(" c ", "C"));
    }
}
