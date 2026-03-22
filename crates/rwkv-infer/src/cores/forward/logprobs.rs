use std::collections::BTreeSet;
use crate::cores::forward::{Logprob, TokenIdLogprobsConfig, TopLogprob};

/// Builds per-token logprob metadata from post-sampling probabilities.
///
/// `SampledTokenLogprob::logprob` always reports the sampled token's own logprob.
/// `top_logprobs` returns the union of the top-N candidates, the sampled token
/// when top-N output is requested, and any explicit resolved candidate token ids.
pub fn build_sampled_token_logprob(
    probs: &[f32],
    sampled_token_id: i32,
    cfg: &TokenIdLogprobsConfig,
) -> Logprob {
    let sampled_index = sampled_token_id.max(0) as usize;
    let sampled_logprob = probs
        .get(sampled_index)
        .copied()
        .map(prob_to_logprob)
        .unwrap_or(f32::NEG_INFINITY);

    let mut ranked: Vec<TopLogprob> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, prob)| *prob > 0.0)
        .map(|(token_id, prob)| TopLogprob {
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

    let mut top_logprobs: Vec<TopLogprob> = ranked
        .into_iter()
        .filter(|candidate| included_token_ids.remove(&candidate.token_id))
        .collect();

    top_logprobs.extend(included_token_ids.into_iter().filter_map(|token_id| {
        probs
            .get(token_id.max(0) as usize)
            .copied()
            .map(|prob| TopLogprob {
                token_id,
                logprob: prob_to_logprob(prob),
            })
    }));

    sort_logprob_candidates(&mut top_logprobs);

    Logprob {
        logprob: sampled_logprob,
        top_logprobs,
    }
}

fn sort_logprob_candidates(candidates: &mut [TopLogprob]) {
    candidates.sort_by(|left, right| {
        right
            .logprob
            .total_cmp(&left.logprob)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
}

fn prob_to_logprob(prob: f32) -> f32 {
    if prob > 0.0 {
        prob.ln()
    } else {
        f32::NEG_INFINITY
    }
}