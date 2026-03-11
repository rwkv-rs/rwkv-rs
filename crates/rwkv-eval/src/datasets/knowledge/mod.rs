use async_openai::Client;
use async_openai::config::OpenAIConfig;
use crate::datasets::{apply_user_assistant_template, CoTMode, SamplingConfig};
use crate::inferers::{CompletionRequest, CompletionResponse};

pub mod ceval;
pub mod cmmlu;
pub mod gpqa;
pub mod mmlu;
pub mod mmlu_pro;
pub mod mmlu_redux;
pub mod mmmlu;
pub mod supergpqa;


pub fn get_expected_context(
    subject: &String,
    question: &String,
    choices: &Vec<String>,
    cot_mode: CoTMode
) -> String {
    let choices = choices.iter().enumerate()
        .map(|(i, choice)| format!("{}. {}", char::from(b'A' + i as u8), choice))
        .collect::<Vec<_>>()
        .join("\n");

    let user_part = format!(
        concat!(
        "User: You are a very talented expert in {subject}.\n",
        "Answer this question and finish with a single option letter.\n",
        "Question: {question}\n",
        "Choices:\n{choices}\n\n",
        ),
        subject = subject,
        question = question,
        choices = choices,
    );

    let assistant_part = match cot_mode {
        CoTMode::NoCoT => "Assistant: Therefore, the answer is<|logprobs_of_choices|>",
        CoTMode::FakeCoT => concat!(
        "Assistant: <think>\n</think>\n",
        "Therefore, the answer is<|logprobs_of_choices|>",
        ),
        CoTMode::CoT => concat!(
        "Assistant: <think><|completions_of_cot|></think>\n",
        "Therefore, the answer is<|logprobs_of_choices|>"
        ),
    }.to_string();

    apply_user_assistant_template(user_part, assistant_part)
}


pub fn get_ref_answer(answer: &u8) -> String {
    char::from(b'A' + answer).to_string()
}


pub async fn get_final_answer(
    model_client: &Client<OpenAIConfig>,
    model_name: &String,
    choices: &Vec<String>,
    prompt_for_final_answer: &String,
    sampling_config: &SamplingConfig,
) -> u8 {
    let choice_token_texts = (0..choices.len())
        .map(|i| format!(" {}", char::from(b'A' + i as u8)))
        .collect::<Vec<_>>();

    let req = CompletionRequest::new(
        model_name.clone(),
        prompt_for_final_answer.into(),
        vec![],
        1,
        &sampling_config,
        Some(1),
        Some(choice_token_texts.clone()),
    );

    let resp: CompletionResponse = model_client.completions().create_byot(&req).await.unwrap();

    let choice_logprobs = resp.choices[0].logprobs.as_ref()
        .and_then(|logprobs| logprobs.top_logprobs.first()).unwrap();

    choice_token_texts.iter().enumerate().max_by(|(_, left), (_, right)| {
        choice_logprobs.get(left.as_str()).copied().unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&choice_logprobs.get(right.as_str()).copied().unwrap_or(f32::NEG_INFINITY))
        }).map(|(idx, _)| idx).unwrap() as u8
}