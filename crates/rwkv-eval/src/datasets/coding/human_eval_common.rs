use crate::datasets::CoTMode;

pub(crate) fn get_prompt(prompt: &str, cot_mode: CoTMode, need_code_prefix: bool) -> String {
    let prompt = trim_empty_lines(prompt);
    let assistant_part = match cot_mode {
        CoTMode::NoCoT if need_code_prefix => "<think>\n</think>\n```python".to_string(),
        CoTMode::NoCoT => prompt.clone(),
        CoTMode::FakeCoT if need_code_prefix => "<think>\n</think>\n```python".to_string(),
        CoTMode::FakeCoT => format!("<think>\n</think>\n{prompt}"),
        CoTMode::CoT if need_code_prefix => {
            "<think><|completions_of_cot|></think>\n```python".to_string()
        }
        CoTMode::CoT => format!("<think><|completions_of_cot|></think>\n{prompt}"),
    };

    format!(
        concat!(
            "User: You are a top-level code master. Complete the following code ",
            "without any additional text or explanation:\n",
            "{prompt}\n\n",
            "Assistant: {assistant_part}"
        ),
        prompt = prompt,
        assistant_part = assistant_part,
    )
}

pub(crate) fn get_check_script(
    program: &str,
    test: &str,
    entry_point: &str,
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

PROGRAM = {program}
TEST = {test}
ENTRY_POINT = {entry_point}
TIMEOUT = {timeout_secs}

{helpers}

try:
    namespace = {{}}
    normalized_test = "".join(TEST.split())
    with time_limit(TIMEOUT):
        exec(PROGRAM, namespace)
    with time_limit(TIMEOUT):
        exec(TEST, namespace)

    candidate = namespace.get(ENTRY_POINT)
    if candidate is None:
        raise AssertionError(f"Missing entry point: {{ENTRY_POINT}}")

    if f"check({{ENTRY_POINT}})" not in normalized_test:
        if "check" not in namespace:
            raise AssertionError("Missing `check` function in test suite")
        with time_limit(TIMEOUT):
            namespace["check"](candidate)
except TimeoutException:
    emit(False, "timed out")
except BaseException as exc:
    emit(False, format_exception(exc))
else:
    emit(True, "")
"#,
        imports = imports,
        program = serde_json::to_string(program).unwrap(),
        test = serde_json::to_string(test).unwrap(),
        entry_point = serde_json::to_string(entry_point).unwrap(),
        timeout_secs = timeout_secs,
        helpers = helpers,
    )
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
