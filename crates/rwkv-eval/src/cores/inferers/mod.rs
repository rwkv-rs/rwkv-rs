use std::collections::BTreeMap;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        chat::{Prompt, StopConfiguration},
        completions::CreateCompletionRequest as BaseCompletionRequest,
    },
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::cores::datasets::SamplingConfig;

#[derive(Clone, Debug, Serialize)]
pub struct CompletionRequest {
    #[serde(flatten)]
    base: BaseCompletionRequest,
    top_k: i32,
    penalty_decay: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_token_texts: Option<Vec<String>>,
}

impl CompletionRequest {
    pub fn new(
        model_name: String,
        prompt: Prompt,
        stop_suffix: Vec<String>,
        max_tokens: u32,
        sampling_config: &SamplingConfig,
        logprobs: Option<u8>,
        candidate_token_texts: Option<Vec<String>>,
    ) -> Self {
        Self {
            base: BaseCompletionRequest {
                model: model_name,
                prompt,
                max_tokens: Some(max_tokens),
                temperature: Some(sampling_config.temperature),
                top_p: Some(sampling_config.top_p),
                stop: Some(StopConfiguration::StringArray(stop_suffix)),
                presence_penalty: Some(sampling_config.presence_penalty),
                frequency_penalty: Some(sampling_config.repetition_penalty),
                logprobs,
                stream: Some(true),
                ..Default::default()
            },
            top_k: sampling_config.top_k,
            penalty_decay: sampling_config.penalty_decay,
            candidate_token_texts,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionResponseChoice>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionResponseChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    pub matched_stop_suffix: Option<String>,
    pub matched_stop_suffix_index: Option<usize>,
    pub generated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionLogprobs {
    pub tokens: Option<Vec<String>>,
    pub token_logprobs: Option<Vec<Option<f32>>>,
    pub top_logprobs: Option<Vec<BTreeMap<String, f32>>>,
    pub text_offset: Option<Vec<usize>>,
}

pub async fn create_completion_streamed(
    model_client: &Client<OpenAIConfig>,
    req: &CompletionRequest,
) -> Result<CompletionResponse, String> {
    let mut stream = model_client
        .completions()
        .create_stream_byot(req)
        .await
        .map_err(|err| format!("completion request failed: {err}"))?;
    let mut meta = None;
    let mut choices = BTreeMap::<u32, CompletionResponseChoice>::new();

    while let Some(chunk) = stream.next().await {
        let chunk: CompletionResponse =
            chunk.map_err(|err| format!("completion stream failed: {err}"))?;
        if meta.is_none() {
            meta = Some((chunk.id.clone(), chunk.created, chunk.model.clone()));
        }
        merge_completion_chunk(&mut choices, chunk);
    }

    let (id, created, model) =
        meta.ok_or_else(|| "completion stream returned no chunks".to_string())?;

    Ok(CompletionResponse {
        id,
        object: "text_completion".to_string(),
        created,
        model,
        choices: choices.into_values().collect(),
    })
}

fn merge_completion_chunk(
    response: &mut BTreeMap<u32, CompletionResponseChoice>,
    chunk: CompletionResponse,
) {
    for chunk_choice in chunk.choices {
        let choice =
            response
                .entry(chunk_choice.index)
                .or_insert_with(|| CompletionResponseChoice {
                    text: String::new(),
                    index: chunk_choice.index,
                    finish_reason: None,
                    matched_stop_suffix: None,
                    matched_stop_suffix_index: None,
                    generated_tokens: None,
                    logprobs: None,
                });
        choice.text.push_str(&chunk_choice.text);
        choice.finish_reason = chunk_choice.finish_reason.or(choice.finish_reason.take());
        choice.matched_stop_suffix = chunk_choice
            .matched_stop_suffix
            .or(choice.matched_stop_suffix.take());
        choice.matched_stop_suffix_index = chunk_choice
            .matched_stop_suffix_index
            .or(choice.matched_stop_suffix_index.take());
        choice.generated_tokens = chunk_choice
            .generated_tokens
            .or(choice.generated_tokens.take());
        merge_logprobs(&mut choice.logprobs, chunk_choice.logprobs);
    }
}

fn merge_logprobs(target: &mut Option<CompletionLogprobs>, incoming: Option<CompletionLogprobs>) {
    let Some(incoming) = incoming else {
        return;
    };

    match target {
        Some(target) => {
            append_optional_vec(&mut target.tokens, incoming.tokens);
            append_optional_vec(&mut target.token_logprobs, incoming.token_logprobs);
            append_optional_vec(&mut target.top_logprobs, incoming.top_logprobs);
            append_optional_vec(&mut target.text_offset, incoming.text_offset);
        }
        None => *target = Some(incoming),
    }
}

fn append_optional_vec<T>(target: &mut Option<Vec<T>>, incoming: Option<Vec<T>>) {
    if let Some(mut incoming) = incoming {
        if let Some(target) = target {
            target.append(&mut incoming);
        } else {
            *target = Some(incoming);
        }
    }
}
