use std::path::Path;

use rwkv_eval::datasets::CoTMode;

pub(crate) fn build_task_log_path(
    logs_root: &Path,
    experiment_name: &str,
    benchmark_name: &str,
    model_name: &str,
    cot_mode: CoTMode,
    n_shot: u8,
    avg_k: f32,
) -> String {
    logs_root
        .join(sanitize_path_component(experiment_name))
        .join(sanitize_path_component(benchmark_name))
        .join(format!(
            "{}_{}_nshot{}_avgk{}.log",
            sanitize_path_component(model_name),
            cot_mode_name(cot_mode).to_ascii_lowercase(),
            n_shot,
            sanitize_path_component(&format!("{avg_k}")),
        ))
        .display()
        .to_string()
}

pub(crate) fn cot_mode_name(cot_mode: CoTMode) -> &'static str {
    match cot_mode {
        CoTMode::NoCoT => "NoCoT",
        CoTMode::FakeCoT => "FakeCoT",
        CoTMode::CoT => "CoT",
    }
}

fn sanitize_path_component(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
            rendered.push(ch);
        } else {
            rendered.push('_');
        }
    }
    rendered.trim_matches('_').to_string()
}
