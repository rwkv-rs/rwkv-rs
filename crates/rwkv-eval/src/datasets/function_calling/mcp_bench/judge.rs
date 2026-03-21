use super::runtime::{normalize_api_base, run_python_wire};
use super::types::{McpBenchEvaluation, McpBenchJudgeRequest, McpBenchJudgeWire, OpenAiLikeConfig};
use crate::init::runtime_ext_api_config_overrides;
use rwkv_config::validated::eval::EVAL_CFG;
use std::path::Path;

pub const MCP_BENCH_PASS_THRESHOLD: f64 = 7.0;

const EVALUATE_SCRIPT: &str = r###"
import asyncio
import json
import os
import sys
import traceback

async def main() -> None:
    request = json.load(sys.stdin)
    try:
        sys.path.insert(0, os.getcwd())
        from benchmark.evaluator import TaskEvaluator
        from llm.provider import LLMProvider
        from openai import AsyncOpenAI

        judge = request["judge_config"]
        client = AsyncOpenAI(api_key=judge["api_key"], base_url=judge["base_url"])
        provider = LLMProvider(client=client, deployment_name=judge["model"], provider_type="openai_compatible")

        evaluator = TaskEvaluator(provider, enable_judge_stability=False)
        evaluation = await evaluator.evaluate(
            task=request["task"],
            execution_results=request["execution_results"],
            final_solution=request["final_solution"],
            total_rounds=request["total_rounds"],
            available_tools=request["available_tools"],
            planning_json_compliance=request["planning_json_compliance"],
            accumulated_information=request.get("accumulated_information", ""),
            concrete_task_description=request.get("concrete_task_description") or None,
            dependency_analysis=request.get("dependency_analysis") or None,
        )

        print(json.dumps({"ok": True, "evaluation": evaluation}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }, ensure_ascii=False))

asyncio.run(main())
"###;

pub fn resolve_judge_config() -> Result<OpenAiLikeConfig, String> {
    let eval_cfg = EVAL_CFG
        .get()
        .ok_or_else(|| "EVAL_CFG is not initialized".to_string())?;
    let runtime_overrides = runtime_ext_api_config_overrides();
    let judger_cfg = runtime_overrides
        .llm_judger
        .unwrap_or_else(|| eval_cfg.llm_judger.clone());
    Ok(OpenAiLikeConfig {
        base_url: normalize_api_base(&judger_cfg.base_url),
        api_key: judger_cfg.api_key,
        model: judger_cfg.model,
    })
}

pub async fn evaluate_with_official_evaluator(
    runtime_root: &Path,
    request: &McpBenchJudgeRequest,
) -> Result<McpBenchEvaluation, String> {
    let wire: McpBenchJudgeWire =
        run_python_wire(runtime_root, EVALUATE_SCRIPT, Some(request)).await?;
    if !wire.ok {
        return Err(if wire.error.is_empty() {
            "official MCP-Bench evaluator failed".to_string()
        } else {
            wire.error
        });
    }
    wire.evaluation
        .ok_or_else(|| "official MCP-Bench evaluator returned no evaluation payload".to_string())
}

pub fn collapse_evaluation_to_pass(evaluation: &McpBenchEvaluation) -> bool {
    evaluation.task_completion_score >= MCP_BENCH_PASS_THRESHOLD
        && evaluation.tool_selection_score >= MCP_BENCH_PASS_THRESHOLD
        && evaluation.planning_effectiveness_and_efficiency_score >= MCP_BENCH_PASS_THRESHOLD
}

pub fn summarize_evaluation(evaluation: &McpBenchEvaluation) -> String {
    format!(
        concat!(
            "task_completion_score={:.2}, ",
            "tool_selection_score={:.2}, ",
            "planning_effectiveness_and_efficiency_score={:.2}, ",
            "valid_tool_name_rate={}, execution_success_rate={}, planning_json_compliance={}"
        ),
        evaluation.task_completion_score,
        evaluation.tool_selection_score,
        evaluation.planning_effectiveness_and_efficiency_score,
        evaluation
            .valid_tool_name_rate
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .execution_success_rate
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .planning_json_compliance
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "n/a".to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::{MCP_BENCH_PASS_THRESHOLD, collapse_evaluation_to_pass};
    use crate::datasets::function_calling::mcp_bench::types::McpBenchEvaluation;

    #[test]
    fn collapse_requires_all_three_aggregate_scores() {
        let passing = McpBenchEvaluation {
            task_completion_score: MCP_BENCH_PASS_THRESHOLD,
            tool_selection_score: 8.0,
            planning_effectiveness_and_efficiency_score: 9.0,
            task_fulfillment: 0.0,
            grounding: 0.0,
            tool_appropriateness: 0.0,
            parameter_accuracy: 0.0,
            dependency_awareness: 0.0,
            parallelism_and_efficiency: 0.0,
            input_schema_compliance: Some(1.0),
            valid_tool_name_rate: Some(1.0),
            execution_success_rate: Some(1.0),
            planning_json_compliance: Some(1.0),
        };
        assert!(collapse_evaluation_to_pass(&passing));

        let failing = McpBenchEvaluation {
            tool_selection_score: 6.9,
            ..passing
        };
        assert!(!collapse_evaluation_to_pass(&failing));
    }
}
