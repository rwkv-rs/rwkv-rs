use crate::datasets::CoTMode;

pub(crate) fn get_prompt(
    prompt: &str,
    reference_code: &str,
    cot_mode: CoTMode,
    need_code_prefix: bool,
) -> String {
    let prompt = get_signature(reference_code)
        .map(|signature| {
            format!(
                "{prompt}\nFunction signature: {signature}\nWrite the full function definition."
            )
        })
        .unwrap_or_else(|| prompt.to_string());
    let prompt = trim_empty_lines(&prompt);
    let assistant_part = match cot_mode {
        CoTMode::NoCoT if need_code_prefix => "<think></think>\n```python".to_string(),
        CoTMode::NoCoT => prompt.clone(),
        CoTMode::FakeCoT if need_code_prefix => "<think></think>\n```python".to_string(),
        CoTMode::FakeCoT => format!("<think></think>\n{prompt}"),
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

pub(crate) fn get_assertion_script(
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
        completion = serde_json::to_string(completion).unwrap(),
        test_imports = serde_json::to_string(test_imports).unwrap(),
        test_list = serde_json::to_string(test_list).unwrap(),
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
