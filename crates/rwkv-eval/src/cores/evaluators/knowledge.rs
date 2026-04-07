use crate::datasets::{CoTMode, SamplingConfig};
use crate::evaluators::{get_completions_of_cot, get_prompt_for_cot, get_prompt_for_final_answer};
use crate::inferers::{CompletionRequest, create_completion_streamed};
use async_openai::Client;
use async_openai::config::OpenAIConfig;

pub fn get_answer_index(answer: &str) -> u8 {
    let answer = answer.trim();
    let answer = answer
        .chars()
        .next()
        .unwrap_or_else(|| panic!("empty answer letter"));

    match answer {
        'A'..='Z' => answer as u8 - b'A',
        'a'..='z' => answer.to_ascii_uppercase() as u8 - b'A',
        _ => panic!("invalid answer letter: {answer}"),
    }
}

pub async fn get_final_answer_with_cot_mode(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    choices: &[String],
    expected_context: &str,
    sampling_config: &SamplingConfig,
    cot_mode: CoTMode,
) -> u8 {
    match cot_mode {
        CoTMode::CoT => {
            let prompt_for_cot = get_prompt_for_cot(expected_context);
            let completions_of_cot =
                get_completions_of_cot(model_client, model_name, &prompt_for_cot, sampling_config)
                    .await;

            let prompt_for_final_answer =
                get_prompt_for_final_answer(expected_context, Some(&completions_of_cot));

            get_final_answer(
                model_client,
                model_name,
                choices,
                &prompt_for_final_answer,
                sampling_config,
            )
            .await
        }
        CoTMode::FakeCoT | CoTMode::NoCoT => {
            let prompt_for_final_answer = get_prompt_for_final_answer(expected_context, None);

            get_final_answer(
                model_client,
                model_name,
                choices,
                &prompt_for_final_answer,
                sampling_config,
            )
            .await
        }
    }
}

async fn get_final_answer(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    choices: &[String],
    prompt_for_final_answer: &str,
    sampling_config: &SamplingConfig,
) -> u8 {
    let choice_token_texts = (0..choices.len())
        .map(|i| format!(" {}", char::from(b'A' + i as u8)))
        .collect::<Vec<_>>();

    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt_for_final_answer.into(),
        vec![],
        1,
        sampling_config,
        Some(1),
        Some(choice_token_texts.clone()),
    );

    let resp = create_completion_streamed(model_client, &req).await.unwrap();

    let choice_logprobs = resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|logprobs| logprobs.top_logprobs.as_ref())
        .and_then(|top_logprobs| top_logprobs.first())
        .unwrap();

    choice_token_texts
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            choice_logprobs
                .get(left.as_str())
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                .total_cmp(
                    &choice_logprobs
                        .get(right.as_str())
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY),
                )
        })
        .map(|(idx, _)| idx)
        .unwrap() as u8
}
