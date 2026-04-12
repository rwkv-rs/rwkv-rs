use async_openai::{Client, config::OpenAIConfig};

use crate::cores::{
    datasets::{
        CoTMode,
        SamplingConfig,
        apply_user_assistant_template,
        get_completions_of_cot,
        get_prompt_for_cot,
        get_prompt_for_final_answer,
        render_context,
    },
    inferers::{CompletionRequest, CompletionResponse, create_completion_streamed},
};

pub mod ceval;
pub mod cmmlu;
pub mod gpqa;
pub mod mmlu;
pub mod mmlu_pro;
pub mod mmlu_redux;
pub mod mmmlu;
pub mod supergpqa;

pub struct Example {
    pub question: String,
    pub choices: Vec<String>,
    pub answer_index: u8,
}

pub struct ChoiceRecord {
    pub context: String,
    pub answer_index: u8,
    pub answer_text: String,
}

fn concat_choices(choices: &[String]) -> String {
    choices
        .iter()
        .enumerate()
        .map(|(i, choice)| format!("{}. {}", char::from(b'A' + i as u8), choice))
        .collect::<Vec<_>>()
        .join("\n")
}

fn concat_examples(examples: &[Example], cot_mode: CoTMode) -> String {
    examples
        .iter()
        .map(|example| {
            let user_part = format!(
                concat!(
                    "You are a very talented expert.\n",
                    "Answer this question and finish with a single option letter.\n",
                    "Question: {question}\n",
                    "Choices:\n{choices}\n\n",
                ),
                question = example.question,
                choices = concat_choices(&example.choices),
            );
            let assistant_part = match cot_mode {
                CoTMode::NoCoT => format!(
                    "Therefore, the answer is {}.\n\n",
                    char::from(b'A' + example.answer_index)
                ),
                CoTMode::FakeCoT => format!(
                    concat!("<think>\n</think>\n", "Therefore, the answer is {}.\n\n",),
                    char::from(b'A' + example.answer_index)
                ),
                _ => panic!("CoT Mode with fewShot is not supported!"),
            }
            .to_string();
            apply_user_assistant_template(user_part, assistant_part)
        })
        .collect::<Vec<_>>()
        .concat()
}

pub fn get_expect_context(
    subject: &str,
    question: &str,
    choices: &[String],
    cot_mode: CoTMode,
    examples: &[Example],
) -> String {
    let examples_part = concat_examples(examples, cot_mode);

    let user_part = format!(
        concat!(
            "You are a very talented expert in {subject}.\n",
            "Answer this question and finish with a single option letter.\n",
            "Question: {question}\n",
            "Choices:\n{choices}",
        ),
        subject = subject,
        question = question,
        choices = concat_choices(choices),
    );

    let assistant_part = match cot_mode {
        CoTMode::NoCoT => "Therefore, the answer is<|logprobs_of_choices|>",
        CoTMode::FakeCoT => concat!(
            "<think>\n</think>\n",
            "Therefore, the answer is<|logprobs_of_choices|>",
        ),
        CoTMode::CoT => concat!(
            "<think><|completions_of_cot|></think>\n",
            "Therefore, the answer is<|logprobs_of_choices|>"
        ),
    }
    .to_string();

    format!(
        "{}\n\n{}",
        examples_part,
        apply_user_assistant_template(user_part, assistant_part)
    )
}

pub fn get_ref_answer(answer_index: u8) -> String {
    char::from(b'A' + answer_index).to_string()
}

pub fn answer_index_from_letter(answer: &str) -> u8 {
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

fn answer_letter(answer_index: u8) -> String {
    char::from(b'A' + answer_index).to_string()
}

pub async fn get_final_answer_with_cot_mode(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    choices: &[String],
    expected_context: &str,
    sampling_config: &SamplingConfig,
    cot_mode: CoTMode,
) -> ChoiceRecord {
    match cot_mode {
        CoTMode::CoT => {
            let prompt_for_cot = get_prompt_for_cot(expected_context);
            let completions_of_cot =
                get_completions_of_cot(model_client, model_name, &prompt_for_cot, sampling_config)
                    .await;

            let prompt_for_final_answer =
                get_prompt_for_final_answer(expected_context, Some(&completions_of_cot));
            let answer_index = get_final_answer(
                model_client,
                model_name,
                choices,
                &prompt_for_final_answer,
                sampling_config,
            )
            .await;
            let answer_text = answer_letter(answer_index);

            ChoiceRecord {
                context: render_context(
                    expected_context,
                    &[
                        ("<|completions_of_cot|>", &completions_of_cot),
                        ("<|logprobs_of_choices|>", &format!(" {}", answer_text)),
                    ],
                ),
                answer_index,
                answer_text,
            }
        }
        CoTMode::FakeCoT | CoTMode::NoCoT => {
            let prompt_for_final_answer = get_prompt_for_final_answer(expected_context, None);
            let answer_index = get_final_answer(
                model_client,
                model_name,
                choices,
                &prompt_for_final_answer,
                sampling_config,
            )
            .await;
            let answer_text = answer_letter(answer_index);

            ChoiceRecord {
                context: render_context(
                    expected_context,
                    &[("<|logprobs_of_choices|>", &format!(" {}", answer_text))],
                ),
                answer_index,
                answer_text,
            }
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
        &sampling_config,
        Some(1),
        Some(choice_token_texts.clone()),
    );

    let resp: CompletionResponse = create_completion_streamed(model_client, &req)
        .await
        .unwrap();

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
