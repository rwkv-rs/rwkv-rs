use super::tau_bench_common::{
    TauDomain, TauTask, create_domain_env, evaluate_task, render_system_prompt, render_user_prompt,
};
use super::{
    FunctionCall, FunctionCallingDecision, FunctionCallingStep, build_turn_completion_prompt,
    get_expected_context, get_completion, parse_tool_call_or_final_answer, render_fc_output,
};
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use sonic_rs::{Value, prelude::*};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

const TAU_BENCH_MAX_STEPS: usize = 16;
const TAU_BENCH_EXPECTED_LEN: usize = 278;

#[distributed_slice(ALL_BENCHMARKS)]
static TAU_BENCH_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("tau_bench"),
    field: Field::FunctionCalling,
    display_name: "tau_bench",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 0.3,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(TauBench::new(dataset_root)),
};

pub struct TauBench {
    dataset_root: PathBuf,
    test: Vec<TauBenchItem>,
}

pub struct TauBenchItem {
    domain: TauDomain,
    task: TauTask,
    system_prompt: String,
    user_prompt: String,
}

impl TauBench {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }

    fn load_domain_items(&self, domain: TauDomain) -> Result<Vec<TauBenchItem>, String> {
        let domain_root = self.dataset_root.join("tau_bench").join(domain.as_str());
        let tasks_path = domain_root.join("tasks.json");
        let split_path = domain_root.join("split_tasks.json");
        if !tasks_path.is_file() || !split_path.is_file() {
            return Err(format!(
                "missing tau_bench files for {} under {}",
                domain.as_str(),
                domain_root.display()
            ));
        }

        let tasks = sonic_rs::from_str::<Vec<TauTask>>(
            &fs::read_to_string(&tasks_path).map_err(|err| err.to_string())?,
        )
        .map_err(|err| format!("failed to parse {}: {err}", tasks_path.display()))?;
        let split_value = sonic_rs::from_str::<Value>(
            &fs::read_to_string(&split_path).map_err(|err| err.to_string())?,
        )
        .map_err(|err| format!("failed to parse {}: {err}", split_path.display()))?;
        let base_ids = split_value
            .get(&"base")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("{} missing base split", split_path.display()))?
            .iter()
            .filter_map(|value| value.as_str())
            .map(str::to_string)
            .collect::<BTreeSet<_>>();

        let env = create_domain_env(domain, &self.dataset_root)?;
        let system_prompt =
            render_system_prompt(env.policy(), env.assistant_tools(), env.user_tools());

        Ok(tasks
            .into_iter()
            .filter(|task| base_ids.contains(&task.id))
            .map(|task| TauBenchItem {
                domain,
                user_prompt: render_user_prompt(&task),
                system_prompt: system_prompt.clone(),
                task,
            })
            .collect())
    }
}

#[async_trait]
impl Benchmark for TauBench {
    fn load(&mut self) -> bool {
        self.test.clear();
        for domain in [TauDomain::Airline, TauDomain::Retail, TauDomain::Telecom] {
            match self.load_domain_items(domain) {
                Ok(mut items) => self.test.append(&mut items),
                Err(_) => return true,
            }
        }
        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.len() != TAU_BENCH_EXPECTED_LEN
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            "tau_bench",
            &[
                UrlDownloadFile {
                    relative_path: PathBuf::from("airline/db.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/db.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("airline/policy.md"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/policy.md".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("airline/tasks.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/tasks.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("airline/split_tasks.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/split_tasks.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("retail/db.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/retail/db.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("retail/policy.md"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/retail/policy.md".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("retail/tasks.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/retail/tasks.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("retail/split_tasks.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/retail/split_tasks.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("telecom/db.toml"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/db.toml".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("telecom/user_db.toml"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/user_db.toml".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("telecom/main_policy.md"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/main_policy.md".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("telecom/tech_support_manual.md"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/tech_support_manual.md".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("telecom/tasks.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/tasks.json".to_string(),
                },
                UrlDownloadFile {
                    relative_path: PathBuf::from("telecom/split_tasks.json"),
                    url: "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/split_tasks.json".to_string(),
                },
            ],
            8,
        ));
        println!("tau_bench dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(cot_mode, CoTMode::CoT, "tau_bench only supports CoT");
        assert_eq!(n_shot, 0, "tau_bench only supports 0-shot");
        let item = &self.test[index];
        get_expected_context(&item.system_prompt, &item.user_prompt, &[])
    }

    fn get_ref_answer(&self, index: usize) -> String {
        let item = &self.test[index];
        let reward_basis = item
            .task
            .evaluation_criteria
            .as_ref()
            .map(|criteria| {
                criteria
                    .reward_basis
                    .iter()
                    .map(|basis| format!("{basis:?}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_else(|| "None".to_string());
        format!(
            "domain={}\ntask_id={}\nreward_basis={}",
            item.domain.as_str(),
            item.task.id,
            reward_basis
        )
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
    ) -> bool {
        if cot_mode != CoTMode::CoT || n_shot != 0 {
            return false;
        }
        let item = &self.test[index];
        let mut env = match create_domain_env(item.domain, &self.dataset_root) {
            Ok(env) => env,
            Err(_) => return false,
        };
        if let Some(initial_state) = &item.task.initial_state {
            if let Some(data) = &initial_state.initialization_data {
                if let Some(agent_data) = &data.agent_data {
                    if env.update_agent_data(agent_data).is_err() {
                        return false;
                    }
                }
                if let Some(user_data) = &data.user_data {
                    if env.update_user_data(user_data).is_err() {
                        return false;
                    }
                }
            }
            for action in initial_state
                .initialization_actions
                .as_deref()
                .unwrap_or(&[])
            {
                if env.run_env_function(action).is_err() {
                    return false;
                }
            }
        }

        let mut steps = Vec::<FunctionCallingStep>::new();
        let mut tool_calls = Vec::<FunctionCall>::new();
        let mut assistant_messages = Vec::<String>::new();

        for _ in 0..TAU_BENCH_MAX_STEPS {
            let cot_context =
                get_expected_context(&item.system_prompt, &item.user_prompt, &steps);
            let cot = get_completion(
                model_client,
                model_name,
                &cot_context,
                &TAU_BENCH_INFO.sampling_config,
                vec!["</think>".to_string()],
                2048,
            )
            .await;
            let decision_prompt = build_turn_completion_prompt(&cot_context, &cot);
            let response = get_completion(
                model_client,
                model_name,
                &decision_prompt,
                &TAU_BENCH_INFO.sampling_config,
                vec!["\nUser:".to_string(), "\nAssistant:".to_string()],
                2048,
            )
            .await;

            match parse_tool_call_or_final_answer(&response) {
                Ok(FunctionCallingDecision::ToolCall(tool_call)) => {
                    let outcome = env.execute_tool_call(&tool_call);
                    let fc_output = render_fc_output(&tool_call, outcome);
                    tool_calls.push(tool_call.clone());
                    steps.push(FunctionCallingStep {
                        cot,
                        tool_call,
                        fc_output,
                    });
                }
                Ok(FunctionCallingDecision::FinalAnswer(answer)) => {
                    assistant_messages.push(answer);
                    break;
                }
                Err(_) => return false,
            }
        }

        if assistant_messages.is_empty() {
            return false;
        }

        evaluate_task(
            item.domain,
            &self.dataset_root,
            &item.task,
            &tool_calls,
            &assistant_messages,
        )
        .unwrap_or(false)
    }
}
