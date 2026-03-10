use crate::inference_core::{RequestedTokenLogprobsConfig, SamplingConfig};

const DEFAULT_TEMPERATURE: f32 = 1.0;
const MIN_TEMPERATURE: f32 = 0.001;
const MAX_TEMPERATURE: f32 = 1000.0;
const DEFAULT_TOP_K: i32 = 0;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;
const DEFAULT_PRESENCE_PENALTY: f32 = 0.0;
const DEFAULT_REPETITION_PENALTY: f32 = 0.0;
const DEFAULT_PENALTY_DECAY: f32 = 1.0;
const MAX_COMPLETION_LOGPROBS: u8 = 5;
const MAX_CHAT_TOP_LOGPROBS: u8 = 20;

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_sampling_config(
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    max_new_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    penalty_decay: Option<f32>,
) -> crate::Result<SamplingConfig> {
    let temperature = temperature.unwrap_or(DEFAULT_TEMPERATURE);
    if !temperature.is_finite() || !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temperature) {
        return Err(crate::Error::bad_request(format!(
            "temperature must be finite and in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}], got {temperature}"
        )));
    }

    let top_k = top_k.unwrap_or(DEFAULT_TOP_K);
    if top_k < 0 {
        return Err(crate::Error::bad_request(format!(
            "top_k must be >= 0, got {top_k}"
        )));
    }

    let top_p = top_p.unwrap_or(DEFAULT_TOP_P);
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(crate::Error::bad_request(format!(
            "top_p must be finite and in [0, 1], got {top_p}"
        )));
    }

    let max_new_tokens = max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    if max_new_tokens < 1 {
        return Err(crate::Error::bad_request(format!(
            "max_new_tokens must be >= 1, got {max_new_tokens}"
        )));
    }

    let presence_penalty = presence_penalty.unwrap_or(DEFAULT_PRESENCE_PENALTY);
    validate_finite("presence_penalty", presence_penalty)?;

    let repetition_penalty = repetition_penalty.unwrap_or(DEFAULT_REPETITION_PENALTY);
    validate_finite("repetition_penalty", repetition_penalty)?;

    let penalty_decay = penalty_decay.unwrap_or(DEFAULT_PENALTY_DECAY);
    validate_finite("penalty_decay", penalty_decay)?;

    Ok(SamplingConfig {
        temperature,
        top_k,
        top_p,
        max_new_tokens: max_new_tokens as usize,
        presence_penalty,
        repetition_penalty,
        penalty_decay,
    })
}

fn validate_finite(name: &str, value: f32) -> crate::Result<()> {
    if !value.is_finite() {
        return Err(crate::Error::bad_request(format!(
            "{name} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn normalize_candidate_token_texts(
    candidate_token_texts: Option<Vec<String>>,
) -> crate::Result<Option<Vec<String>>> {
    let Some(candidate_token_texts) = candidate_token_texts else {
        return Ok(None);
    };

    let mut deduped = Vec::with_capacity(candidate_token_texts.len());
    for token_text in candidate_token_texts {
        if token_text.is_empty() {
            return Err(crate::Error::bad_request(
                "candidate_token_texts cannot contain empty strings",
            ));
        }
        if !deduped.contains(&token_text) {
            deduped.push(token_text);
        }
    }

    if deduped.is_empty() {
        Ok(None)
    } else {
        Ok(Some(deduped))
    }
}

pub(crate) fn validate_completion_logprobs(
    logprobs: Option<u8>,
    candidate_token_texts: Option<Vec<String>>,
) -> crate::Result<Option<RequestedTokenLogprobsConfig>> {
    let candidate_token_texts = normalize_candidate_token_texts(candidate_token_texts)?;
    let Some(logprobs) = logprobs else {
        if candidate_token_texts.is_some() {
            return Err(crate::Error::bad_request(
                "candidate_token_texts requires logprobs to be set",
            ));
        }
        return Ok(None);
    };

    if logprobs > MAX_COMPLETION_LOGPROBS {
        return Err(crate::Error::bad_request(format!(
            "logprobs must be in [0, {MAX_COMPLETION_LOGPROBS}], got {logprobs}"
        )));
    }
    if candidate_token_texts.is_some() && logprobs == 0 {
        return Err(crate::Error::bad_request(
            "candidate_token_texts requires logprobs >= 1",
        ));
    }

    Ok(Some(RequestedTokenLogprobsConfig {
        top_logprobs: logprobs as usize,
        candidate_token_texts,
    }))
}

pub(crate) fn validate_chat_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    candidate_token_texts: Option<Vec<String>>,
) -> crate::Result<Option<RequestedTokenLogprobsConfig>> {
    let logprobs_enabled = logprobs.unwrap_or(false);
    let candidate_token_texts = normalize_candidate_token_texts(candidate_token_texts)?;

    if let Some(top_logprobs) = top_logprobs {
        if !logprobs_enabled {
            return Err(crate::Error::bad_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        if top_logprobs > MAX_CHAT_TOP_LOGPROBS {
            return Err(crate::Error::bad_request(format!(
                "top_logprobs must be in [0, {MAX_CHAT_TOP_LOGPROBS}], got {top_logprobs}"
            )));
        }
    }

    if !logprobs_enabled {
        if candidate_token_texts.is_some() {
            return Err(crate::Error::bad_request(
                "candidate_token_texts requires logprobs=true",
            ));
        }
        return Ok(None);
    }

    let top_logprobs = top_logprobs.unwrap_or(0);
    if candidate_token_texts.is_some() && top_logprobs == 0 {
        return Err(crate::Error::bad_request(
            "candidate_token_texts requires top_logprobs >= 1",
        ));
    }

    Ok(Some(RequestedTokenLogprobsConfig {
        top_logprobs: top_logprobs as usize,
        candidate_token_texts,
    }))
}
