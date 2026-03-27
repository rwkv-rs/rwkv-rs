use std::path::{Path, PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use sonic_rs::Value;

use super::get_expected_context;
use crate::{
    datasets::{
        ALL_BENCHMARKS,
        Benchmark,
        BenchmarkInfo,
        BenchmarkName,
        CoTMode,
        Field,
        Record,
        SamplingConfig,
        coding::{extract_code, get_code_completion_with_cot_mode},
        utils::hf::downloader::{UrlDownloadFile, download_url_files},
    },
    evaluators::coding::run_python_verdict_script,
};

#[distributed_slice(ALL_BENCHMARKS)]
static MBPP_PLUS_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("mbpp_plus"),
    field: Field::Coding,
    display_name: "MBPP+",
    cot_mode: &[CoTMode::NoCoT, CoTMode::FakeCoT, CoTMode::CoT],
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
    create: |dataset_root| Box::new(MbppPlus::new(dataset_root)),
};

pub struct MbppPlus {
    dataset_root: PathBuf,
    test: Vec<MbppPlusItem>,
}

pub struct MbppPlusItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    base_input: String,
    plus_input: String,
    atol: f64,
}

#[derive(Debug, Deserialize)]
struct RawMbppPlusItem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    base_input: Value,
    plus_input: Value,
    atol: Option<f64>,
}

fn read_mbpp_plus_items<P: AsRef<Path>>(path: P) -> Vec<RawMbppPlusItem> {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use flate2::read::GzDecoder;

    let file = File::open(path.as_ref()).unwrap();
    let reader = BufReader::new(GzDecoder::new(file));

    reader
        .lines()
        .map(|line| line.unwrap())
        .filter(|line| !line.trim().is_empty())
        .filter(|line| !(line.contains("Infinity") || line.contains("-Infinity")))
        .map(|line| sonic_rs::from_str(&line).unwrap())
        .collect()
}

impl MbppPlus {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for MbppPlus {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("mbpp_plus")
            .join("MbppPlus-v0.2.0.jsonl.gz");
        if !path.is_file() {
            return true;
        }

        self.test = read_mbpp_plus_items(path)
            .into_iter()
            .map(|item| MbppPlusItem {
                task_id: item.task_id,
                prompt: item.prompt,
                entry_point: item.entry_point,
                canonical_solution: item.canonical_solution,
                base_input: sonic_rs::to_string(&item.base_input).unwrap(),
                plus_input: sonic_rs::to_string(&item.plus_input).unwrap(),
                atol: item.atol.unwrap_or(0.0),
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
            "mbpp_plus",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("MbppPlus-v0.2.0.jsonl.gz"),
                url: "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
                    .to_string(),
            }],
            1,
        ).await;
        println!("mbpp_plus dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, _n_shot: u8) -> String {
        let item = &self.test[index];
        get_expected_context(&item.prompt, &item.canonical_solution, cot_mode)
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
    ) -> Record {
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_code_completion_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &MBPP_PLUS_INFO.sampling_config,
            cot_mode,
            1024,
        )
        .await;
        let answer = extract_code(&generated.completion);
        let verdict = run_python_verdict_script(&get_judge_script(
            &item.task_id,
            &answer,
            &item.entry_point,
            &item.canonical_solution,
            &item.base_input,
            &item.plus_input,
            item.atol,
            3,
        ))
        .await
        .unwrap_or_else(|err| {
            panic!(
                "mbpp_plus sandbox execution failed: {err}; task={}",
                item.task_id
            )
        });

        let ref_answer = self.get_ref_answer(index);
        Record {
            context: generated.context,
            answer,
            ref_answer,
            is_passed: verdict.passed,
            fail_reason: if verdict.passed {
                String::new()
            } else {
                verdict.fail_reason
            },
        }
    }
}

fn get_judge_script(
    task_id: &str,
    completion: &str,
    entry_point: &str,
    canonical_solution: &str,
    base_input: &str,
    plus_input: &str,
    atol: f64,
    timeout_secs: i32,
) -> String {
    let imports = "import contextlib\nimport json\nimport signal\nimport copy";
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

MBPP_OUTPUT_NOT_NONE_TASKS = {"check_str", "text_match_three", "text_starta_endb"}
MBPP_OUTPUT_SET_EQ_TASKS = {
    "similar_elements",
    "find_char_long",
    "common_in_nested_lists",
    "extract_singly",
    "larg_nnum",
    "intersection_array",
    "find_dissimilar",
    "Diff",
}

def deserialize_inputs(task_id, inputs):
    task_id = int(str(task_id).split("/")[-1])
    if task_id in [
        2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429, 470,
        560, 579, 596, 616, 630, 726, 740, 744, 809,
    ]:
        return [[tuple(lst) for lst in inp] for inp in inputs]
    if task_id in [63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780]:
        return [[[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs]
    if task_id in [75, 413, 444, 753]:
        return [[[tuple(lst) for lst in inp[0]]] + [inp[1]] for inp in inputs]
    if task_id in [106, 750]:
        return [[inp[0]] + [tuple(inp[1])] for inp in inputs]
    if task_id == 115:
        return [[[
            set(item) if isinstance(item, list) and len(item) else {}
            for item in inp[0]
        ]] for inp in inputs]
    if task_id == 124:
        return [(float(inp[0]), complex(inp[1])) for inp in inputs]
    if task_id in [250, 405, 446, 617, 720, 763, 808]:
        return [[tuple(inp[0])] + [inp[1]] for inp in inputs]
    if task_id in [259, 401, 445]:
        modified = [[[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs]
        return [[tuple(lst) for lst in inp] for inp in modified]
    if task_id == 278:
        modified = [[[tuple(item) if isinstance(item, list) else item for item in inp[0]]] for inp in inputs]
        return [[tuple(lst) for lst in inp] for inp in modified]
    if task_id == 307:
        return [[tuple(inp[0])] + [inp[1], inp[2]] for inp in inputs]
    if task_id == 722:
        return [[{key: tuple(value) for key, value in inp[0].items()}] + inp[1:] for inp in inputs]
    if task_id == 252:
        return [[complex(inp[0])] for inp in inputs]
    if task_id in [580, 615, 791]:
        def turn_all_list_into_tuple(inp):
            if isinstance(inp, list):
                return tuple(turn_all_list_into_tuple(item) for item in inp)
            return inp
        return [turn_all_list_into_tuple(inp) for inp in inputs]
    return inputs

def is_equal(actual, expected):
    if isinstance(actual, set) and isinstance(expected, set):
        return actual == expected
    if isinstance(actual, (int, float, complex)) and isinstance(expected, (int, float, complex)):
        if isinstance(actual, complex) or isinstance(expected, complex):
            return actual == expected
        return abs(float(actual) - float(expected)) <= ATOL
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(is_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(is_equal(actual[key], expected[key]) for key in actual)
    return actual == expected
"#
    .trim();

    format!(
        r#"{imports}

TASK_ID = {task_id}
COMPLETION = {completion}
ENTRY_POINT = {entry_point}
CANONICAL_SOLUTION = {canonical_solution}
BASE_INPUT = json.loads({base_input})
PLUS_INPUT = json.loads({plus_input})
ATOL = {atol}
TIMEOUT = {timeout_secs}

{helpers}

try:
    reference_ns = {{}}
    candidate_ns = {{}}

    with time_limit(TIMEOUT):
        exec(CANONICAL_SOLUTION, reference_ns)
    with time_limit(TIMEOUT):
        exec(COMPLETION, candidate_ns)

    reference = reference_ns.get(ENTRY_POINT)
    candidate = candidate_ns.get(ENTRY_POINT)
    if reference is None or candidate is None:
        raise AssertionError(f"Missing entry point: {{ENTRY_POINT}}")

    for sample in deserialize_inputs(TASK_ID, BASE_INPUT) + deserialize_inputs(TASK_ID, PLUS_INPUT):
        args = sample if isinstance(sample, (list, tuple)) else [sample]
        with time_limit(TIMEOUT):
            expected = reference(*copy.deepcopy(args))
        with time_limit(TIMEOUT):
            actual = candidate(*copy.deepcopy(args))

        if ENTRY_POINT in MBPP_OUTPUT_NOT_NONE_TASKS:
            if actual is None:
                emit(False, "failed: Output should not be None")
                break
            continue

        if ENTRY_POINT in MBPP_OUTPUT_SET_EQ_TASKS:
            if set(actual) != set(expected):
                emit(False, "failed: Wrong Answer")
                break
            continue

        if not is_equal(actual, expected):
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
        task_id = sonic_rs::to_string(task_id).unwrap(),
        completion = sonic_rs::to_string(completion).unwrap(),
        entry_point = sonic_rs::to_string(entry_point).unwrap(),
        canonical_solution = sonic_rs::to_string(canonical_solution).unwrap(),
        base_input = sonic_rs::to_string(base_input).unwrap(),
        plus_input = sonic_rs::to_string(plus_input).unwrap(),
        atol = atol,
        timeout_secs = timeout_secs,
        helpers = helpers,
    )
}
