use sonic_rs::{json, to_string};
use xgrammar::{
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLTensor,
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    get_bitmask_shape,
    reset_token_bitmask,
};

use super::SamplingConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintSpec {
    pub schema_json: String,
    pub strict_mode: bool,
}

#[derive(Debug)]
pub struct ConstraintSample {
    pub token_id: i32,
    pub probs: Vec<f32>,
    pub finish_after_token: bool,
}

pub struct ConstraintState {
    matcher: GrammarMatcher,
    token_bitmask: Box<[i32]>,
    bitmask_shape: Vec<i64>,
    bitmask_strides: Vec<i64>,
    penalties: Vec<f32>,
    rng_state: Option<u32>,
    vocab_size: usize,
}

impl std::fmt::Debug for ConstraintState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintState")
            .field("token_bitmask_len", &self.token_bitmask.len())
            .field("bitmask_shape", &self.bitmask_shape)
            .field("bitmask_strides", &self.bitmask_strides)
            .field("penalties_len", &self.penalties.len())
            .field("rng_state", &self.rng_state)
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

// Safety: the matcher is only owned and mutated by the single execution-loop task.
// We never share references to it across threads or access it concurrently.
unsafe impl Send for ConstraintState {}

impl ConstraintState {
    pub fn new(
        spec: &ConstraintSpec,
        compiler: &mut GrammarCompiler,
        vocab_size: usize,
    ) -> crate::Result<Self> {
        let compiled = compiler
            .compile_json_schema(
                &spec.schema_json,
                true,
                None,
                None::<(&str, &str)>,
                spec.strict_mode,
                None,
            )
            .map_err(crate::Error::bad_request)?;
        let matcher =
            GrammarMatcher::new(&compiled, None, true, -1).map_err(crate::Error::bad_request)?;
        let token_bitmask = xgrammar::allocate_token_bitmask(1, vocab_size);
        let (_, bitmask_size) = get_bitmask_shape(1, vocab_size);

        Ok(Self {
            matcher,
            token_bitmask,
            bitmask_shape: vec![1, bitmask_size as i64],
            bitmask_strides: vec![bitmask_size as i64, 1],
            penalties: vec![0.0; vocab_size],
            rng_state: None,
            vocab_size,
        })
    }

    pub fn sample(
        &mut self,
        logits: &[f32],
        sampling: &SamplingConfig,
        seed: u32,
    ) -> crate::Result<ConstraintSample> {
        let mut sampling = *sampling;
        sampling.check(self.vocab_size as i32);
        if logits.len() < self.vocab_size {
            return Err(crate::Error::internal(format!(
                "constraint sampler logits shorter than vocab: {} < {}",
                logits.len(),
                self.vocab_size
            )));
        }
        if self.matcher.is_terminated() {
            return Err(crate::Error::internal(
                "constraint matcher terminated before sampling".to_string(),
            ));
        }

        let needs_apply = self.fill_next_token_bitmask()?;
        let mut scaled_logits = vec![f32::NEG_INFINITY; self.vocab_size];
        let mut max_scaled = f32::NEG_INFINITY;

        for (token_id, scaled_logit) in scaled_logits.iter_mut().enumerate() {
            if needs_apply && !bitmask_accepts(&self.token_bitmask, token_id) {
                continue;
            }
            let penalty = self.penalties[token_id];
            let raw = logits[token_id] - penalty;
            let value = sanitize_f32(raw * (1.0 / sampling.temperature));
            *scaled_logit = value;
            if value > max_scaled {
                max_scaled = value;
            }
        }

        if !max_scaled.is_finite() {
            return Err(crate::Error::internal(
                "constraint grammar rejected every next token".to_string(),
            ));
        }

        let mut probs = vec![0.0f32; self.vocab_size];
        let mut exp_sum = 0.0f32;
        for (token_id, prob) in probs.iter_mut().enumerate() {
            let scaled = scaled_logits[token_id];
            if !scaled.is_finite() {
                continue;
            }
            let value = (scaled - max_scaled).exp();
            *prob = value;
            exp_sum += value;
        }

        if !(exp_sum.is_finite() && exp_sum > 0.0) {
            return Err(crate::Error::internal(
                "constraint sampler failed to normalize probabilities".to_string(),
            ));
        }

        for prob in &mut probs {
            *prob /= exp_sum;
        }

        let mut ranked: Vec<(usize, f32)> = probs
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, prob)| *prob > 0.0)
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        if ranked.is_empty() {
            return Err(crate::Error::internal(
                "constraint sampler found no valid ranked candidates".to_string(),
            ));
        }

        let top_k = match sampling.top_k {
            n if n <= 0 => ranked.len(),
            n => (n as usize).min(ranked.len()),
        };
        let mut selected = ranked[..top_k].to_vec();

        if sampling.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut nucleus = Vec::with_capacity(selected.len());
            for candidate in &selected {
                nucleus.push(*candidate);
                cumulative += candidate.1;
                if cumulative >= sampling.top_p {
                    break;
                }
            }
            if !nucleus.is_empty() {
                selected = nucleus;
            }
        }

        let selected_sum: f32 = selected.iter().map(|(_, prob)| *prob).sum();
        if !(selected_sum.is_finite() && selected_sum > 0.0) {
            return Err(crate::Error::internal(
                "constraint sampler selected zero-probability candidate set".to_string(),
            ));
        }

        probs.fill(0.0);
        for (token_id, prob) in &selected {
            probs[*token_id] = *prob / selected_sum;
        }

        let target = self.next_random(seed) * probs.iter().copied().sum::<f32>();
        let mut cumulative = 0.0f32;
        let mut sampled_token_id = selected[0].0;
        for (token_id, _) in &selected {
            cumulative += probs[*token_id];
            if target <= cumulative {
                sampled_token_id = *token_id;
                break;
            }
        }

        self.update_penalties(sampled_token_id, &sampling);
        if !self.matcher.accept_token(sampled_token_id as i32) {
            return Err(crate::Error::internal(format!(
                "constraint matcher rejected sampled token {sampled_token_id}"
            )));
        }

        Ok(ConstraintSample {
            token_id: sampled_token_id as i32,
            probs,
            finish_after_token: self.matcher.is_terminated(),
        })
    }

    fn fill_next_token_bitmask(&mut self) -> crate::Result<bool> {
        reset_token_bitmask(&mut self.token_bitmask);
        let mut tensor = DLTensor {
            data: self.token_bitmask.as_mut_ptr() as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            },
            shape: self.bitmask_shape.as_mut_ptr(),
            strides: self.bitmask_strides.as_mut_ptr(),
            byte_offset: 0,
        };
        Ok(self.matcher.fill_next_token_bitmask(&mut tensor, 0, false))
    }

    fn next_random(&mut self, seed: u32) -> f32 {
        let rng_state = self.rng_state.get_or_insert(seed.max(1));
        *rng_state = lcg_step(*rng_state);
        u32_to_unit_interval_open(*rng_state)
    }

    fn update_penalties(&mut self, sampled_token_id: usize, sampling: &SamplingConfig) {
        if !sampling.penalties_enabled() {
            return;
        }

        let current_penalty = self.penalties[sampled_token_id];
        for penalty in &mut self.penalties {
            *penalty *= sampling.penalty_decay;
        }
        self.penalties[sampled_token_id] += if current_penalty == 0.0 {
            sampling.presence_penalty
        } else {
            sampling.repetition_penalty
        };
    }
}

pub fn build_tokenizer_info_from_vocab(
    vocab_tokens: &[Vec<u8>],
    vocab_size: usize,
    stop_token_ids: &[i32],
) -> TokenizerInfo {
    let metadata = json!({
        "vocab_type": 0,
        "vocab_size": vocab_size,
        "add_prefix_space": false,
        "stop_token_ids": stop_token_ids,
    });
    TokenizerInfo::from_vocab_and_metadata_bytes(
        vocab_tokens.iter().map(Vec::as_slice),
        &to_string(&metadata).unwrap_or_else(|_| "{}".to_string()),
    )
}

fn sanitize_f32(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else if x.is_infinite() {
        if x.is_sign_negative() {
            -f32::MAX
        } else {
            f32::MAX
        }
    } else {
        x
    }
}

fn lcg_step(z: u32) -> u32 {
    z.wrapping_mul(1664525).wrapping_add(1013904223)
}

fn u32_to_unit_interval_open(int_random: u32) -> f32 {
    let shifted = int_random >> 9;
    (shifted as f32 + 1.0) / 8_388_609.0
}

fn bitmask_accepts(bitmask: &[i32], token_id: usize) -> bool {
    let word_idx = token_id / 32;
    let bit_idx = token_id % 32;
    bitmask
        .get(word_idx)
        .is_some_and(|word| (word & (1 << bit_idx)) != 0)
}
