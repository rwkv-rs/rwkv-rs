use crate::inference_core::{SamplingConfig, TokenLogprobsConfig};

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

fn normalize_candidate_token_ids(
    candidate_token_ids: Option<Vec<i32>>,
    vocab_size: usize,
) -> crate::Result<Option<Vec<i32>>> {
    let Some(candidate_token_ids) = candidate_token_ids else {
        return Ok(None);
    };

    let mut deduped = Vec::with_capacity(candidate_token_ids.len());
    for token_id in candidate_token_ids {
        if token_id < 0 || token_id as usize >= vocab_size {
            return Err(crate::Error::bad_request(format!(
                "candidate_token_ids contains invalid token id {token_id}; expected [0, {vocab_size})"
            )));
        }
        if !deduped.contains(&token_id) {
            deduped.push(token_id);
        }
    }

    Ok(Some(deduped))
}

pub(crate) fn validate_completion_logprobs(
    logprobs: Option<u8>,
    candidate_token_ids: Option<Vec<i32>>,
    vocab_size: usize,
) -> crate::Result<Option<TokenLogprobsConfig>> {
    let candidate_token_ids = normalize_candidate_token_ids(candidate_token_ids, vocab_size)?;
    let Some(logprobs) = logprobs else {
        if candidate_token_ids.is_some() {
            return Err(crate::Error::bad_request(
                "candidate_token_ids requires logprobs to be set",
            ));
        }
        return Ok(None);
    };

    if logprobs > MAX_COMPLETION_LOGPROBS {
        return Err(crate::Error::bad_request(format!(
            "logprobs must be in [0, {MAX_COMPLETION_LOGPROBS}], got {logprobs}"
        )));
    }
    if candidate_token_ids.is_some() && logprobs == 0 {
        return Err(crate::Error::bad_request(
            "candidate_token_ids requires logprobs >= 1",
        ));
    }

    Ok(Some(TokenLogprobsConfig {
        top_logprobs: logprobs as usize,
        candidate_token_ids,
    }))
}

pub(crate) fn validate_chat_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    candidate_token_ids: Option<Vec<i32>>,
    vocab_size: usize,
) -> crate::Result<Option<TokenLogprobsConfig>> {
    let logprobs_enabled = logprobs.unwrap_or(false);
    let candidate_token_ids = normalize_candidate_token_ids(candidate_token_ids, vocab_size)?;

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
        if candidate_token_ids.is_some() {
            return Err(crate::Error::bad_request(
                "candidate_token_ids requires logprobs=true",
            ));
        }
        return Ok(None);
    }

    let top_logprobs = top_logprobs.unwrap_or(0);
    if candidate_token_ids.is_some() && top_logprobs == 0 {
        return Err(crate::Error::bad_request(
            "candidate_token_ids requires top_logprobs >= 1",
        ));
    }

    Ok(Some(TokenLogprobsConfig {
        top_logprobs: top_logprobs as usize,
        candidate_token_ids,
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MAX_NEW_TOKENS, MAX_TEMPERATURE, MIN_TEMPERATURE, validate_chat_logprobs,
        validate_completion_logprobs, validate_sampling_config,
    };

    const TEST_VOCAB_SIZE: usize = 8;

    fn valid() -> crate::Result<crate::inference_core::SamplingConfig> {
        validate_sampling_config(None, None, None, None, None, None, None)
    }

    #[test]
    fn defaults_are_valid() {
        let cfg = valid().expect("default config should be valid");
        assert_eq!(cfg.max_new_tokens, DEFAULT_MAX_NEW_TOKENS as usize);
    }

    #[test]
    fn accepts_temperature_boundaries() {
        validate_sampling_config(Some(MIN_TEMPERATURE), None, None, None, None, None, None)
            .expect("min temperature should be accepted");
        validate_sampling_config(Some(MAX_TEMPERATURE), None, None, None, None, None, None)
            .expect("max temperature should be accepted");
    }

    #[test]
    fn rejects_invalid_temperature() {
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = validate_sampling_config(Some(value), None, None, None, None, None, None)
                .expect_err("temperature should be rejected");
            assert!(err.to_string().contains("temperature"));
        }
    }

    #[test]
    fn rejects_invalid_top_p() {
        for value in [-0.1, 1.1, f32::NAN] {
            let err = validate_sampling_config(None, None, Some(value), None, None, None, None)
                .expect_err("top_p should be rejected");
            assert!(err.to_string().contains("top_p"));
        }
    }

    #[test]
    fn rejects_invalid_top_k() {
        let err = validate_sampling_config(None, Some(-1), None, None, None, None, None)
            .expect_err("negative top_k should be rejected");
        assert!(err.to_string().contains("top_k"));
    }

    #[test]
    fn rejects_zero_max_new_tokens() {
        let err = validate_sampling_config(None, None, None, Some(0), None, None, None)
            .expect_err("zero max_new_tokens should be rejected");
        assert!(err.to_string().contains("max_new_tokens"));
    }

    #[test]
    fn rejects_non_finite_penalties() {
        for (presence, repetition, decay) in [
            (f32::NAN, 0.0, 1.0),
            (0.0, f32::INFINITY, 1.0),
            (0.0, 0.0, f32::NEG_INFINITY),
        ] {
            let err = validate_sampling_config(
                None,
                None,
                None,
                None,
                Some(presence),
                Some(repetition),
                Some(decay),
            )
            .expect_err("non-finite penalties should be rejected");
            assert!(
                err.to_string().contains("presence_penalty")
                    || err.to_string().contains("repetition_penalty")
                    || err.to_string().contains("penalty_decay")
            );
        }
    }

    #[test]
    fn validates_completion_logprobs_range() {
        assert_eq!(
            validate_completion_logprobs(Some(5), None, TEST_VOCAB_SIZE)
                .expect("logprobs=5 should pass")
                .expect("config should exist")
                .top_logprobs,
            5
        );

        let err = validate_completion_logprobs(Some(6), None, TEST_VOCAB_SIZE)
            .expect_err("logprobs=6 must fail");
        assert!(err.to_string().contains("logprobs"));
    }

    #[test]
    fn validates_chat_logprobs_dependencies() {
        assert!(validate_chat_logprobs(Some(true), Some(20), None, TEST_VOCAB_SIZE).is_ok());

        let err = validate_chat_logprobs(Some(false), Some(1), None, TEST_VOCAB_SIZE)
            .expect_err("top_logprobs without logprobs=true must fail");
        assert!(
            err.to_string()
                .contains("top_logprobs requires logprobs=true")
        );

        let err = validate_chat_logprobs(Some(true), Some(21), None, TEST_VOCAB_SIZE)
            .expect_err("top_logprobs above 20 must fail");
        assert!(err.to_string().contains("top_logprobs"));
    }

    #[test]
    fn validates_candidate_token_ids() {
        let cfg = validate_completion_logprobs(Some(1), Some(vec![3, 1, 3, 7]), TEST_VOCAB_SIZE)
            .expect("candidate ids should pass")
            .expect("config should exist");
        assert_eq!(cfg.candidate_token_ids, Some(vec![3, 1, 7]));

        let err = validate_completion_logprobs(Some(1), Some(vec![8]), TEST_VOCAB_SIZE)
            .expect_err("out-of-range candidate id must fail");
        assert!(err.to_string().contains("candidate_token_ids"));
    }

    #[test]
    fn candidate_token_ids_require_logprobs() {
        let err = validate_completion_logprobs(None, Some(vec![1]), TEST_VOCAB_SIZE)
            .expect_err("candidate_token_ids without logprobs must fail");
        assert!(
            err.to_string()
                .contains("candidate_token_ids requires logprobs to be set")
        );

        let err = validate_completion_logprobs(Some(0), Some(vec![1]), TEST_VOCAB_SIZE)
            .expect_err("candidate_token_ids with zero completion logprobs must fail");
        assert!(
            err.to_string()
                .contains("candidate_token_ids requires logprobs >= 1")
        );

        let err = validate_chat_logprobs(Some(false), None, Some(vec![1]), TEST_VOCAB_SIZE)
            .expect_err("candidate_token_ids without chat logprobs must fail");
        assert!(
            err.to_string()
                .contains("candidate_token_ids requires logprobs=true")
        );

        let err = validate_chat_logprobs(Some(true), None, Some(vec![1]), TEST_VOCAB_SIZE)
            .expect_err("candidate_token_ids without chat top_logprobs must fail");
        assert!(
            err.to_string()
                .contains("candidate_token_ids requires top_logprobs >= 1")
        );

        let err = validate_chat_logprobs(Some(true), Some(0), Some(vec![1]), TEST_VOCAB_SIZE)
            .expect_err("candidate_token_ids with zero chat top_logprobs must fail");
        assert!(
            err.to_string()
                .contains("candidate_token_ids requires top_logprobs >= 1")
        );
    }
}
