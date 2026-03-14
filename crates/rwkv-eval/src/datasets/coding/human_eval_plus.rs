use super::human_eval_common::get_expected_context;
use crate::datasets::coding::{extract_code, get_code_completion_with_cot_mode};
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_gzip_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::evaluators::coding::run_python_verdict_script;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static HUMAN_EVAL_PLUS_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("human_eval_plus"),
    field: Field::Coding,
    display_name: "HumanEval+",
    cot_mode: &[CoTMode::NoCoT],
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
    with_llm_judger: false,
    create: |dataset_root| Box::new(HumanEvalPlus::new(dataset_root)),
};

pub struct HumanEvalPlus {
    dataset_root: PathBuf,
    test: Vec<HumanEvalPlusItem>,
}

pub struct HumanEvalPlusItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    contract: String,
    base_input: String,
    plus_input: String,
    atol: f64,
}

#[derive(Debug, Deserialize)]
struct RawHumanEvalPlusItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    contract: Option<String>,
    base_input: Value,
    plus_input: Value,
    atol: Option<f64>,
}

impl HumanEvalPlus {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn get_judge_script(
    prompt: &str,
    completion: &str,
    entry_point: &str,
    canonical_solution: &str,
    contract: &str,
    base_input: &str,
    plus_input: &str,
    atol: f64,
    timeout_secs: i32,
) -> String {
    let imports = "import contextlib\nimport json\nimport signal";
    let helpers = r#"
class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def emit(passed, fail_reason):
    print(json.dumps({"passed": passed, "fail_reason": fail_reason}, ensure_ascii=False))

def format_exception(exc, include_traceback=False):
    exc_type = type(exc).__name__
    exc_msg = str(exc).strip()
    header = f"failed: {exc_type}: {exc_msg}" if exc_msg else f"failed: {exc_type}"
    if not include_traceback:
        return header
    import traceback
    tb = traceback.format_exc().rstrip()
    return f"{header}\n{tb}" if tb else header

def is_close(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= ATOL
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(is_close(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(is_close(actual[key], expected[key]) for key in actual)
    return actual == expected
"#
    .trim();

    format!(
        r#"{imports}

PROMPT = {prompt}
COMPLETION = {completion}
ENTRY_POINT = {entry_point}
CANONICAL_SOLUTION = {canonical_solution}
CONTRACT = {contract}
BASE_INPUT = json.loads({base_input})
PLUS_INPUT = json.loads({plus_input})
ATOL = {atol}
TIMEOUT = {timeout_secs}

{helpers}

try:
    reference_ns = {{}}
    candidate_ns = {{}}
    reference_program = PROMPT + CONTRACT + CANONICAL_SOLUTION
    candidate_program = PROMPT + COMPLETION

    with time_limit(TIMEOUT):
        exec(reference_program, reference_ns)
    with time_limit(TIMEOUT):
        exec(candidate_program, candidate_ns)

    reference = reference_ns.get(ENTRY_POINT)
    candidate = candidate_ns.get(ENTRY_POINT)
    if reference is None or candidate is None:
        raise AssertionError(f"Missing entry point: {{ENTRY_POINT}}")

    for sample in BASE_INPUT + PLUS_INPUT:
        args = sample if isinstance(sample, list) else [sample]
        with time_limit(TIMEOUT):
            expected = reference(*args)
        with time_limit(TIMEOUT):
            actual = candidate(*args)
        if not is_close(actual, expected):
            emit(False, "failed: Wrong Answer")
            break
    else:
        emit(True, "")
except TimeoutException:
    emit(False, "timed out")
except BaseException as exc:
    emit(False, format_exception(exc))
"#,
        imports = imports,
        prompt = serde_json::to_string(prompt).unwrap(),
        completion = serde_json::to_string(completion).unwrap(),
        entry_point = serde_json::to_string(entry_point).unwrap(),
        canonical_solution = serde_json::to_string(canonical_solution).unwrap(),
        contract = serde_json::to_string(contract).unwrap(),
        base_input = serde_json::to_string(base_input).unwrap(),
        plus_input = serde_json::to_string(plus_input).unwrap(),
        atol = atol,
        timeout_secs = timeout_secs,
        helpers = helpers,
    )
}

#[async_trait]
impl Benchmark for HumanEvalPlus {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("human_eval_plus")
            .join("HumanEvalPlus-v0.1.9.jsonl.gz");
        if !path.is_file() {
            return true;
        }

        self.test = read_gzip_jsonl_items::<RawHumanEvalPlusItem, _>(path)
            .into_iter()
            .map(|item| HumanEvalPlusItem {
                task_id: item.task_id,
                prompt: item.prompt,
                entry_point: item.entry_point,
                canonical_solution: item.canonical_solution,
                contract: item.contract.unwrap_or_default(),
                base_input: serde_json::to_string(&item.base_input).unwrap(),
                plus_input: serde_json::to_string(&item.plus_input).unwrap(),
                atol: item.atol.unwrap_or(0.0),
            })
            .collect();

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.is_empty()
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            "human_eval_plus",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("HumanEvalPlus-v0.1.9.jsonl.gz"),
                url: "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.9/HumanEvalPlus.jsonl.gz"
                    .to_string(),
            }],
            1,
        ));
        println!("human_eval_plus dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        if cot_mode != CoTMode::NoCoT {
            panic!("human_eval_plus only supports NoCoT, got {cot_mode:?}");
        }

        let item = &self.test[index];
        get_expected_context(&item.prompt, Some(item.prompt.as_str()), cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].canonical_solution.clone()
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
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let completion = get_code_completion_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &HUMAN_EVAL_PLUS_INFO.sampling_config,
            cot_mode,
            1024,
        )
        .await;
        let completion = extract_code(&completion);
        let verdict = run_python_verdict_script(&get_judge_script(
            &item.prompt,
            &completion,
            &item.entry_point,
            &item.canonical_solution,
            &item.contract,
            &item.base_input,
            &item.plus_input,
            item.atol,
            3,
        ))
        .await
        .unwrap_or_else(|err| {
            panic!(
                "human_eval_plus sandbox execution failed: {err}; task={}",
                item.task_id
            )
        });

        verdict.passed
    }
}
