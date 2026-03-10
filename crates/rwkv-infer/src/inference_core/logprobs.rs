use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequestedTokenLogprobsConfig {
    pub top_logprobs: usize,
    pub candidate_token_texts: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenLogprobsConfig {
    pub top_logprobs: usize,
    pub candidate_token_ids: Option<Vec<i32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledTokenTopLogprob {
    pub token_id: i32,
    pub logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledTokenLogprob {
    pub logprob: f32,
    pub top_logprobs: Vec<SampledTokenTopLogprob>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledToken {
    pub batch_index: usize,
    pub token_id: i32,
    pub logprob: Option<SampledTokenLogprob>,
}

fn prob_to_logprob(prob: f32) -> f32 {
    if prob > 0.0 {
        prob.ln()
    } else {
        f32::NEG_INFINITY
    }
}

fn sort_logprob_candidates(candidates: &mut [SampledTokenTopLogprob]) {
    candidates.sort_by(|left, right| {
        right
            .logprob
            .total_cmp(&left.logprob)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
}

/// Builds per-token logprob metadata from post-sampling probabilities.
///
/// `SampledTokenLogprob::logprob` always reports the sampled token's own logprob.
/// `top_logprobs` returns the union of the top-N candidates, the sampled token
/// when top-N output is requested, and any explicit resolved candidate token ids.
pub fn build_sampled_token_logprob(
    probs: &[f32],
    sampled_token_id: i32,
    cfg: &TokenLogprobsConfig,
) -> SampledTokenLogprob {
    let sampled_index = sampled_token_id.max(0) as usize;
    let sampled_logprob = probs
        .get(sampled_index)
        .copied()
        .map(prob_to_logprob)
        .unwrap_or(f32::NEG_INFINITY);

    let mut ranked: Vec<SampledTokenTopLogprob> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, prob)| *prob > 0.0)
        .map(|(token_id, prob)| SampledTokenTopLogprob {
            token_id: token_id as i32,
            logprob: prob_to_logprob(prob),
        })
        .collect();
    sort_logprob_candidates(&mut ranked);

    let mut included_token_ids = BTreeSet::new();
    included_token_ids.extend(
        ranked
            .iter()
            .take(cfg.top_logprobs)
            .map(|candidate| candidate.token_id),
    );

    if cfg.top_logprobs > 0 && sampled_token_id >= 0 && sampled_index < probs.len() {
        included_token_ids.insert(sampled_token_id);
    }

    if let Some(candidate_token_ids) = cfg.candidate_token_ids.as_ref() {
        included_token_ids.extend(candidate_token_ids.iter().copied());
    }

    let mut top_logprobs: Vec<SampledTokenTopLogprob> = ranked
        .into_iter()
        .filter(|candidate| included_token_ids.remove(&candidate.token_id))
        .collect();

    top_logprobs.extend(included_token_ids.into_iter().filter_map(|token_id| {
        probs
            .get(token_id.max(0) as usize)
            .copied()
            .map(|prob| SampledTokenTopLogprob {
                token_id,
                logprob: prob_to_logprob(prob),
            })
    }));

    sort_logprob_candidates(&mut top_logprobs);

    SampledTokenLogprob {
        logprob: sampled_logprob,
        top_logprobs,
    }
}

#[cfg(test)]
mod tests {
    use super::{TokenLogprobsConfig, build_sampled_token_logprob};

    #[test]
    fn returns_sampled_logprob_and_top_candidates() {
        let result = build_sampled_token_logprob(
            &[0.60, 0.20, 0.10, 0.10],
            2,
            &TokenLogprobsConfig {
                top_logprobs: 2,
                candidate_token_ids: None,
            },
        );

        assert!((result.logprob - 0.10f32.ln()).abs() < 1e-6);
        assert_eq!(result.top_logprobs.len(), 3);
        assert_eq!(result.top_logprobs[0].token_id, 0);
        assert_eq!(result.top_logprobs[1].token_id, 1);
        assert_eq!(result.top_logprobs[2].token_id, 2);
    }

    #[test]
    fn candidate_token_ids_are_included_even_when_top_logprobs_is_zero() {
        let result = build_sampled_token_logprob(
            &[0.10, 0.60, 0.20, 0.10],
            1,
            &TokenLogprobsConfig {
                top_logprobs: 0,
                candidate_token_ids: Some(vec![3, 2]),
            },
        );

        assert_eq!(result.top_logprobs.len(), 2);
        assert_eq!(result.top_logprobs[0].token_id, 2);
        assert_eq!(result.top_logprobs[1].token_id, 3);
    }

    #[test]
    fn requested_candidates_keep_zero_probability_entries() {
        let result = build_sampled_token_logprob(
            &[0.70, 0.30, 0.0, 0.0],
            0,
            &TokenLogprobsConfig {
                top_logprobs: 0,
                candidate_token_ids: Some(vec![2, 3]),
            },
        );

        assert_eq!(result.top_logprobs.len(), 2);
        assert_eq!(result.top_logprobs[0].token_id, 2);
        assert_eq!(result.top_logprobs[0].logprob, f32::NEG_INFINITY);
        assert_eq!(result.top_logprobs[1].token_id, 3);
        assert_eq!(result.top_logprobs[1].logprob, f32::NEG_INFINITY);
    }

    #[test]
    fn out_of_range_sampled_token_returns_negative_infinity() {
        let result = build_sampled_token_logprob(
            &[0.25, 0.75],
            99,
            &TokenLogprobsConfig {
                top_logprobs: 1,
                candidate_token_ids: None,
            },
        );

        assert_eq!(result.logprob, f32::NEG_INFINITY);
        assert_eq!(result.top_logprobs.len(), 1);
        assert_eq!(result.top_logprobs[0].token_id, 1);
    }

    #[test]
    fn ties_are_sorted_by_token_id() {
        let result = build_sampled_token_logprob(
            &[0.5, 0.5, 0.0],
            0,
            &TokenLogprobsConfig {
                top_logprobs: 2,
                candidate_token_ids: None,
            },
        );

        assert_eq!(result.top_logprobs[0].token_id, 0);
        assert_eq!(result.top_logprobs[1].token_id, 1);
    }
}
