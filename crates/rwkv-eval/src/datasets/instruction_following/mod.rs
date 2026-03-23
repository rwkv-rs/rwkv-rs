pub mod arena_hard_v2;
pub mod instruction_following;
pub mod wmt24pp;

pub fn sanitize_visible_answer(raw_text: &str, stop_suffixes: &[&str]) -> String {
    let cutoff = stop_suffixes
        .iter()
        .copied()
        .chain(["\nUser:", "\nAssistant:"].into_iter())
        .filter_map(|marker| raw_text.find(marker))
        .min()
        .unwrap_or(raw_text.len());

    let mut visible = raw_text[..cutoff].trim();
    loop {
        let trimmed = visible.trim_start();
        if !trimmed.starts_with("<think>") {
            visible = trimmed;
            break;
        }
        let Some(end) = trimmed.find("</think>") else {
            return String::new();
        };
        visible = &trimmed[end + "</think>".len()..];
    }

    let visible = visible
        .strip_prefix("Assistant:")
        .or_else(|| visible.strip_prefix("User:"))
        .unwrap_or(visible)
        .trim();

    visible
        .replace("<think>", "")
        .replace("</think>", "")
        .trim()
        .to_string()
}
