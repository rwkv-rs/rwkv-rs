use crate::types::SamplingConfig;

const DEFAULT_TEMPERATURE: f32 = 1.0;
const MIN_TEMPERATURE: f32 = 0.001;
const MAX_TEMPERATURE: f32 = 1000.0;
const DEFAULT_TOP_K: i32 = 0;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;
const DEFAULT_PRESENCE_PENALTY: f32 = 0.0;
const DEFAULT_REPETITION_PENALTY: f32 = 0.0;
const DEFAULT_PENALTY_DECAY: f32 = 1.0;

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

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_MAX_NEW_TOKENS, MAX_TEMPERATURE, MIN_TEMPERATURE, validate_sampling_config,
    };

    fn valid() -> crate::Result<crate::types::SamplingConfig> {
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
}
