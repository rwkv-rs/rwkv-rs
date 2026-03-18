use super::get_expected_context;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use rwkv_config::raw::eval::IntApiConfig;
use rwkv_config::validated::eval::EVAL_CFG;
use serde::Deserialize;
use sonic_rs::{JsonValueTrait, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tokio::process::Command as TokioCommand;

const MCP_UNIVERSE_EXPECTED_LEN: usize = 140;
const MCP_UNIVERSE_GIT_URL: &str = "https://github.com/SalesforceAIResearch/MCP-Universe.git";

#[distributed_slice(ALL_BENCHMARKS)]
static MCP_UNIVERSE_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mcp_universe"),
    field: Field::FunctionCalling,
    display_name: "MCP-Universe",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        presence_penalty: 0.0,
        repetition_penalty: 0.0,
        penalty_decay: 1.0,
    },
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(McpUniverse::new(dataset_root)),
};

pub struct McpUniverse {
    dataset_root: PathBuf,
    test: Vec<McpUniverseItem>,
}

pub struct McpUniverseItem {
    domain: McpUniverseDomain,
    task_relative_path: String,
    question: String,
    output_format: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpUniverseDomain {
    WebSearch,
    LocationNavigation,
    FinancialAnalysis,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct McpUniverseWire {
    ok: bool,
    passed: bool,
    answer: String,
    reason: String,
    error: String,
}

impl McpUniverse {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }

    fn runtime_root(&self) -> PathBuf {
        self.dataset_root.join("mcp_universe").join("runtime")
    }

    fn configs_root(&self) -> PathBuf {
        self.runtime_root()
            .join("mcpuniverse")
            .join("benchmark")
            .join("configs")
    }

    fn load_items(&self) -> Result<Vec<McpUniverseItem>, String> {
        let runtime_root = self.runtime_root();
        if !runtime_root.is_dir() {
            return Err(format!(
                "missing MCP-Universe runtime checkout under {}",
                runtime_root.display()
            ));
        }

        let mut items = Vec::new();
        for domain in McpUniverseDomain::ALL {
            for task_relative_path in self.load_domain_task_paths(domain)? {
                let task_path = self.configs_root().join(&task_relative_path);
                let task_text = fs::read_to_string(&task_path)
                    .map_err(|err| format!("failed to read {}: {err}", task_path.display()))?;
                let task = sonic_rs::from_str::<Value>(&task_text)
                    .map_err(|err| format!("failed to parse {}: {err}", task_path.display()))?;
                let question = task
                    .get(&"question")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| format!("{} missing string question", task_path.display()))?
                    .trim()
                    .to_string();
                let output_format = task
                    .get(&"output_format")
                    .map(|value| sonic_rs::to_string_pretty(value).unwrap())
                    .unwrap_or_else(|| "{}".to_string());

                items.push(McpUniverseItem {
                    domain,
                    task_relative_path,
                    question,
                    output_format,
                });
            }
        }

        Ok(items)
    }

    fn load_domain_task_paths(&self, domain: McpUniverseDomain) -> Result<Vec<String>, String> {
        let path = self
            .configs_root()
            .join("mcpuniverse")
            .join(domain.yaml_name());
        let text = fs::read_to_string(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;

        let mut tasks = Vec::new();
        let mut in_tasks = false;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed == "tasks:" {
                in_tasks = true;
                continue;
            }
            if !in_tasks {
                continue;
            }

            if let Some(task) = trimmed.strip_prefix("- ") {
                tasks.push(task.to_string());
                continue;
            }

            if !trimmed.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                break;
            }
        }

        if tasks.is_empty() {
            return Err(format!("{} declared no tasks", path.display()));
        }

        Ok(tasks)
    }
}

impl McpUniverseDomain {
    const ALL: [Self; 3] = [
        Self::WebSearch,
        Self::LocationNavigation,
        Self::FinancialAnalysis,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::WebSearch => "web_search",
            Self::LocationNavigation => "location_navigation",
            Self::FinancialAnalysis => "financial_analysis",
        }
    }

    fn yaml_name(self) -> &'static str {
        match self {
            Self::WebSearch => "web_search.yaml",
            Self::LocationNavigation => "location_navigation.yaml",
            Self::FinancialAnalysis => "financial_analysis.yaml",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Self::WebSearch => concat!(
                "You are an agent for web searching. ",
                "If you don't have enough information to answer the question, ",
                "you can use the google-search tool. ",
                "If you want to obtain the information with a specific URL, ",
                "you can use the fetch tool."
            ),
            Self::LocationNavigation => "You are an agent for location navigation.",
            Self::FinancialAnalysis => concat!(
                "You are an agent for financial analysis. ",
                "If you need to calculate the result, you can use the calculator tool."
            ),
        }
    }
}

#[async_trait]
impl Benchmark for McpUniverse {
    fn load(&mut self) -> bool {
        self.test.clear();
        match self.load_items() {
            Ok(items) => {
                self.test = items;
                self.test.is_empty()
            }
            Err(_) => true,
        }
    }

    async fn check(&self) -> bool {
        self.test.len() != MCP_UNIVERSE_EXPECTED_LEN
            || self
                .test
                .iter()
                .any(|item| item.question.trim().is_empty() || item.output_format.trim().is_empty())
    }

    async fn download(&self) {
        let root = self.dataset_root.join("mcp_universe");
        fs::create_dir_all(&root).unwrap_or_else(|err| {
            panic!(
                "failed to create mcp_universe dataset directory {}: {err}",
                root.display()
            )
        });

        let runtime_root = root.join("runtime");
        let status = if runtime_root.join(".git").is_dir() {
            StdCommand::new("git")
                .arg("-C")
                .arg(&runtime_root)
                .arg("pull")
                .arg("--ff-only")
                .status()
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to update MCP-Universe runtime under {}: {err}",
                        runtime_root.display()
                    )
                })
        } else {
            assert!(
                !runtime_root.exists(),
                "mcp_universe runtime path already exists but is not a git checkout: {}",
                runtime_root.display(),
            );
            StdCommand::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg(MCP_UNIVERSE_GIT_URL)
                .arg(&runtime_root)
                .status()
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to clone MCP-Universe runtime into {}: {err}",
                        runtime_root.display()
                    )
                })
        };

        assert!(
            status.success(),
            "MCP-Universe runtime git command failed with status {}",
            status,
        );
        println!("mcp_universe runtime: {}", runtime_root.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(cot_mode, CoTMode::CoT, "mcp_universe only supports CoT");
        assert_eq!(n_shot, 0, "mcp_universe only supports 0-shot");

        let item = &self.test[index];
        let system_prompt = item.domain.instruction();
        let user_prompt = format!(
            concat!(
                "{question}\n\n",
                "Return output that matches this schema:\n",
                "{output_format}\n\n",
                "Delegated runtime: official MCP-Universe compatibility profile."
            ),
            question = item.question.as_str(),
            output_format = item.output_format.as_str(),
        );
        get_expected_context(system_prompt, &user_prompt, &[])
    }

    fn get_ref_answer(&self, index: usize) -> String {
        let item = &self.test[index];
        format!(
            "domain={}\ntask={}\noutput_format={}",
            item.domain.as_str(),
            item.task_relative_path.as_str(),
            item.output_format.as_str()
        )
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        _model_client: &Client<OpenAIConfig>,
        _judger_model_name: Option<&str>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let expected_context = self.get_expected_context(index, CoTMode::CoT, 0);
        let ref_answer = self.get_ref_answer(index);
        if cot_mode != CoTMode::CoT || n_shot != 0 {
            return Record {
                context: expected_context,
                answer: String::new(),
                ref_answer,
                is_passed: false,
                fail_reason: format!(
                    "unsupported mcp_universe config: cot_mode={cot_mode:?}, n_shot={n_shot}"
                ),
            };
        }

        let item = &self.test[index];
        let api_cfg = match find_eval_api_config(model_name) {
            Ok(api_cfg) => api_cfg,
            Err(err) => {
                return Record {
                    context: expected_context,
                    answer: String::new(),
                    ref_answer,
                    is_passed: false,
                    fail_reason: err,
                };
            }
        };

        let output =
            match run_official_mcp_task(&self.runtime_root(), item, model_name, &api_cfg).await {
                Ok(output) => output,
                Err(err) => {
                    return Record {
                        context: expected_context,
                        answer: String::new(),
                        ref_answer,
                        is_passed: false,
                        fail_reason: err,
                    };
                }
            };

        if !output.ok {
            return Record {
                context: expected_context,
                answer: output.answer,
                ref_answer,
                is_passed: false,
                fail_reason: if output.error.trim().is_empty() {
                    "mcp_universe runtime returned non-ok result".to_string()
                } else {
                    format!("mcp_universe runtime error: {}", output.error.trim())
                },
            };
        }

        Record {
            context: expected_context,
            answer: output.answer,
            ref_answer,
            is_passed: output.passed,
            fail_reason: if output.passed {
                String::new()
            } else if output.reason.trim().is_empty() {
                "official MCP-Universe evaluators marked the task incorrect".to_string()
            } else {
                output.reason.trim().to_string()
            },
        }
    }
}

fn find_eval_api_config(model_name: &str) -> Result<IntApiConfig, String> {
    let eval_cfg = EVAL_CFG
        .get()
        .ok_or_else(|| "rwkv_config::validated::eval::EVAL_CFG is not initialized".to_string())?;
    let matches = eval_cfg
        .models
        .iter()
        .filter(|cfg| cfg.model == model_name)
        .cloned()
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [] => Err(format!(
            "mcp_universe could not find api config for model `{model_name}` in EVAL_CFG.models"
        )),
        [cfg] => Ok(cfg.clone()),
        _ => Err(format!(
            "mcp_universe requires a unique api config for model `{model_name}`, but found {} matches",
            matches.len()
        )),
    }
}

async fn run_official_mcp_task(
    runtime_root: &Path,
    item: &McpUniverseItem,
    model_name: &str,
    api_cfg: &IntApiConfig,
) -> Result<McpUniverseWire, String> {
    let base_url = norm_api_url(&api_cfg.base_url);
    let output = TokioCommand::new("python3")
        .arg("-c")
        .arg(MCP_UNIVERSE_RUNNER_SCRIPT)
        .arg(runtime_root)
        .arg(item.domain.as_str())
        .arg(&item.task_relative_path)
        .arg(model_name)
        .env("OPENAI_API_KEY", &api_cfg.api_key)
        .env("OPENAI_BASE_URL", base_url)
        .env("PYTHONUNBUFFERED", "1")
        .current_dir(runtime_root)
        .output()
        .await
        .map_err(|err| {
            format!("failed to run official MCP-Universe runtime with python3: {err}")
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(wire) = parse_mcp_universe_wire(&stdout) {
        return Ok(wire);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(format!(
        "failed to parse MCP-Universe runtime output; status={}; stderr={stderr}; stdout={stdout}",
        output.status
    ))
}

fn norm_api_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    assert!(!base_url.is_empty(), "base_url cannot be empty");

    let base_url = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("http://{base_url}")
    };
    let base_url = base_url.trim_end_matches('/').to_string();
    if base_url.ends_with("/v1") {
        base_url
    } else {
        format!("{base_url}/v1")
    }
}

fn parse_mcp_universe_wire(stdout: &str) -> Option<McpUniverseWire> {
    stdout
        .lines()
        .rev()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(|line| sonic_rs::from_str::<McpUniverseWire>(line).ok())
}

const MCP_UNIVERSE_RUNNER_SCRIPT: &str = r#"
import asyncio
import json
import os
import sys
import tempfile

runtime_root, domain_key, task_path, model_name = sys.argv[1:5]
sys.path.insert(0, runtime_root)

import yaml
from mcpuniverse.benchmark.runner import BenchmarkRunner

PROFILES = {
    "web_search": {
        "agent_name": "HarmonyReAct-agent",
        "agent_type": "harmony_react",
        "instruction": "You are an agent for web searching. If you don't have enough information to answer the question, you can use the google-search tool. If you want to obtain the information with a specific URL, you can use the fetch tool.",
        "max_iterations": 50,
        "summarize_tool_response": True,
        "servers": [],
    },
    "location_navigation": {
        "agent_name": "ReAct-agent",
        "agent_type": "react",
        "instruction": "You are an agent for location navigation.",
        "max_iterations": 20,
        "summarize_tool_response": False,
        "servers": [],
    },
    "financial_analysis": {
        "agent_name": "FunctionCall-agent",
        "agent_type": "function_call",
        "instruction": "You are an agent for financial analysis. If you need to calculate the result, you can use the calculator tool.",
        "max_iterations": 20,
        "summarize_tool_response": False,
        "servers": [{"name": "yfinance"}, {"name": "calculator"}],
    },
}

async def main():
    temp_path = None
    try:
        profile = PROFILES[domain_key]
        agent_config = {
            "llm": "llm-1",
            "instruction": profile["instruction"],
            "max_iterations": profile["max_iterations"],
            "summarize_tool_response": profile["summarize_tool_response"],
        }
        if profile["servers"]:
            agent_config["servers"] = profile["servers"]
        docs = [
            {
                "kind": "llm",
                "spec": {
                    "name": "llm-1",
                    "type": "openai",
                    "config": {
                        "model_name": model_name,
                    },
                },
            },
            {
                "kind": "agent",
                "spec": {
                    "name": profile["agent_name"],
                    "type": profile["agent_type"],
                    "config": agent_config,
                },
            },
            {
                "kind": "benchmark",
                "spec": {
                    "description": f"rwkv-eval MCP-Universe compatibility run for {task_path}",
                    "agent": profile["agent_name"],
                    "tasks": [task_path],
                },
            },
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as file:
            yaml.safe_dump_all(docs, file, sort_keys=False, allow_unicode=True)
            temp_path = file.name

        benchmark = BenchmarkRunner(temp_path)
        results = await benchmark.run()
        eval_results = results[0].task_results[task_path]["evaluation_results"]
        if not eval_results:
            print(json.dumps({
                "ok": True,
                "passed": False,
                "answer": "",
                "reason": "official runtime returned no evaluation results",
                "error": "",
            }, ensure_ascii=False))
            return

        first_response = eval_results[0].response
        if isinstance(first_response, dict):
            answer = json.dumps(first_response, ensure_ascii=False, sort_keys=True)
        else:
            answer = str(first_response)

        passed = True
        reasons = []
        for result in eval_results:
            if not result.passed:
                passed = False
                reasons.append(result.reason or result.error or "evaluation failed")

        print(json.dumps({
            "ok": True,
            "passed": passed,
            "answer": answer,
            "reason": "; ".join(reasons),
            "error": "",
        }, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "passed": False,
            "answer": "",
            "reason": "",
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

asyncio.run(main())
"#;

#[cfg(test)]
mod tests {
    use super::{McpUniverseWire, norm_api_url, parse_mcp_universe_wire};

    #[test]
    fn normalizes_api_url() {
        assert_eq!(norm_api_url("localhost:1234"), "http://localhost:1234/v1");
        assert_eq!(
            norm_api_url("https://example.com/v1"),
            "https://example.com/v1"
        );
    }

    #[test]
    fn parses_last_json_line() {
        let stdout = "noise\n{\"ok\":true,\"passed\":false,\"answer\":\"x\",\"reason\":\"bad\",\"error\":\"\"}\n";
        let wire = parse_mcp_universe_wire(stdout).unwrap();
        assert_eq!(
            wire,
            McpUniverseWire {
                ok: true,
                passed: false,
                answer: "x".to_string(),
                reason: "bad".to_string(),
                error: String::new(),
            }
        );
    }
}
