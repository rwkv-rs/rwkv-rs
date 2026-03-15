use crate::datasets::{CoTMode, apply_user_assistant_template};

pub fn get_expected_context(prompt: &str, code: &str, cot_mode: CoTMode) -> String {
    let prompt = get_signature(code)
        .map(|signature| {
            format!(
                "{prompt}\nFunction signature: {signature}\nWrite the full function definition."
            )
        })
        .unwrap_or_else(|| prompt.to_string());
    let prompt = trim_empty_lines(&prompt);
    let user_part = format!(
        concat!(
            "You are a top-level code master.\n",
            "{prompt}\n",
            "Output only the full Python function definition without any additional text or explanation."
        ),
        prompt = prompt,
    );

    let assistant_part = match cot_mode {
        CoTMode::NoCoT => "```python\n<|completions|>".to_string(),
        CoTMode::FakeCoT => "<think>\n</think>\n```python\n<|completions|>".to_string(),
        CoTMode::CoT => concat!(
            "<think><|completions_of_cot|></think>\n",
            "```python\n<|completions|>",
        )
        .to_string(),
    }
    .to_string();

    apply_user_assistant_template(user_part, assistant_part)
}

pub fn get_judge_script(
    completion: &str,
    test_imports: &[String],
    test_list: &[String],
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
"#
    .trim();

    format!(
        r#"{imports}

COMPLETION = {completion}
TEST_IMPORTS = json.loads({test_imports})
TEST_LIST = json.loads({test_list})
TIMEOUT = {timeout_secs}

{helpers}

try:
    namespace = {{}}
    with time_limit(TIMEOUT):
        exec("\n".join(TEST_IMPORTS), namespace)
    with time_limit(TIMEOUT):
        exec(COMPLETION, namespace)
    for assertion in TEST_LIST:
        with time_limit(TIMEOUT):
            exec(assertion, namespace)
except TimeoutException:
    emit(False, "timed out")
except BaseException as exc:
    emit(False, format_exception(exc))
else:
    emit(True, "")
"#,
        imports = imports,
        completion = sonic_rs::to_string(completion).unwrap(),
        test_imports = sonic_rs::to_string(test_imports).unwrap(),
        test_list = sonic_rs::to_string(test_list).unwrap(),
        timeout_secs = timeout_secs,
        helpers = helpers,
    )
}

fn get_signature(code: &str) -> Option<String> {
    code.lines().find_map(|line| {
        let stripped = line.trim();
        (stripped.starts_with("def ") && stripped.ends_with(':')).then(|| stripped.to_string())
    })
}

fn trim_empty_lines(prompt: &str) -> String {
    prompt
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}
