mod human_eval_common;
mod mbpp_common;

pub mod human_eval;
pub mod human_eval_cn;
pub mod human_eval_fix;
pub mod human_eval_plus;
pub mod livecodebench;
pub mod mbpp;
pub mod mbpp_plus;

pub fn extract_code(text: &str) -> String {
    if !text.contains("```") {
        return text.trim().to_string();
    }

    let mut last_block = None;
    for part in text.split("```").skip(1).step_by(2) {
        last_block = Some(part);
    }

    let Some(block) = last_block else {
        return text.trim().to_string();
    };

    let mut lines = block.lines();
    let first = lines.next().unwrap_or_default().trim().to_ascii_lowercase();
    if matches!(first.as_str(), "python" | "py") {
        return lines.collect::<Vec<_>>().join("\n").trim().to_string();
    }

    block.trim().to_string()
}
