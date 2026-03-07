use async_trait::async_trait;

use crate::error::EvalError;
use crate::evaluators::EvaluationSampleResult;
use crate::runtime::EvalRuntime;

#[async_trait]
pub trait Checker: Send + Sync {
    async fn classify_error(
        &self,
        runtime: &EvalRuntime,
        benchmark_name: &str,
        result: &EvaluationSampleResult,
    ) -> Result<Option<String>, EvalError>;
}

#[derive(Default)]
pub struct LlmChecker;

#[async_trait]
impl Checker for LlmChecker {
    async fn classify_error(
        &self,
        runtime: &EvalRuntime,
        benchmark_name: &str,
        result: &EvaluationSampleResult,
    ) -> Result<Option<String>, EvalError> {
        if result.is_pass {
            return Ok(None);
        }

        let label = runtime
            .classify_error(
                benchmark_name,
                &result.context,
                &result.ref_answer,
                &result.final_answer,
            )
            .await?;

        Ok(Some(label))
    }
}
