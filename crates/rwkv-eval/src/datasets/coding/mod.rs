use crate::datasets::{
    CoTMode, SamplingConfig, get_completions_of_cot, get_prompt_for_cot, render_context,
};
use crate::evaluators::coding::get_completion;
use async_openai::Client;
use async_openai::config::OpenAIConfig;

mod human_eval_common;
mod mbpp_common;

pub mod human_eval;
pub mod human_eval_cn;
pub mod human_eval_fix;
pub mod human_eval_plus;
pub mod livecodebench;
pub mod mbpp;
pub mod mbpp_plus;

pub struct CodeGeneration {
    pub context: String,
    pub completion: String,
}

pub fn get_prompt_for_code_completion(
    expected_context: &str,
    completions_of_cot: Option<&str>,
) -> String {
    completions_of_cot
        .map(|cot| expected_context.replace("<|completions_of_cot|>", cot))
        .unwrap_or_else(|| expected_context.to_string())
        .split_once("<|completions|>")
        .unwrap_or_else(|| {
            panic!(
                "expected_context missing <|completions|> marker: {}",
                expected_context
            )
        })
        .0
        .to_string()
}

pub async fn get_code_completion_with_cot_mode(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    expected_context: &str,
    sampling_config: &SamplingConfig,
    cot_mode: CoTMode,
    max_tokens: u32,
) -> CodeGeneration {
    let prompt_for_code = match cot_mode {
        CoTMode::CoT => {
            let prompt_for_cot = get_prompt_for_cot(expected_context);
            let completions_of_cot =
                get_completions_of_cot(model_client, model_name, &prompt_for_cot, sampling_config)
                    .await;

            let prompt_for_code =
                get_prompt_for_code_completion(expected_context, Some(&completions_of_cot));
            let completion = get_completion(
                model_client,
                model_name,
                &prompt_for_code,
                sampling_config,
                vec!["```".to_string()],
                max_tokens,
            )
            .await;

            return CodeGeneration {
                context: render_context(
                    expected_context,
                    &[
                        ("<|completions_of_cot|>", &completions_of_cot),
                        ("<|completions|>", &completion),
                    ],
                ),
                completion,
            };
        }
        CoTMode::FakeCoT | CoTMode::NoCoT => get_prompt_for_code_completion(expected_context, None),
    };

    let completion = get_completion(
        model_client,
        model_name,
        &prompt_for_code,
        sampling_config,
        vec!["```".to_string()],
        max_tokens,
    )
    .await;

    CodeGeneration {
        context: render_context(
            expected_context,
            &[("<|completions_of_cot|>", ""), ("<|completions|>", &completion)],
        ),
        completion,
    }
}

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

#[cfg(test)]
mod tests {
    use super::get_prompt_for_code_completion;

    #[test]
    fn prompt_for_code_completion_stops_at_marker() {
        let expected_context = "User: test\n\nAssistant: ```python\nprefix\n<|completions|>tail";
        let prompt = get_prompt_for_code_completion(expected_context, None);
        assert_eq!(prompt, "User: test\n\nAssistant: ```python\nprefix\n");
    }

    #[test]
    fn prompt_for_code_completion_replaces_cot_before_split() {
        let expected_context = concat!(
            "User: test\n\nAssistant: <think><|completions_of_cot|></think>\n",
            "```python\n<|completions|>"
        );
        let prompt = get_prompt_for_code_completion(expected_context, Some("reasoning"));
        assert_eq!(
            prompt,
            "User: test\n\nAssistant: <think>reasoning</think>\n```python\n"
        );
    }
}
