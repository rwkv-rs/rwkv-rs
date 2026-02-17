use rwkv::custom::Tensor;
use rwkv::custom::prelude::{Backend, Int, TensorData};
use rwkv::custom::tensor::DType;
use rwkv::infer::engine::InferExecutor;
use rwkv::infer::{Error, Result, SamplingConfig};

use rwkv::nn::kernels::rapid_sample::{RapidSampleBackend, RapidSamplePenaltyConfig, rapid_sample};
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;

use crate::model::AutoRegressiveModel;

rwkv::custom_mode!();

pub struct RwkvLmExecutor<B: Backend>
where
    B: Wkv7Backend + RapidSampleBackend,
{
    device: B::Device,
    model: AutoRegressiveModel<B>,

    max_batch_size: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>>,
    state: Vec<Tensor<B, 4>>,
    embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>>,

    rng_states: Tensor<B, 1, Int>,
    penalties: Tensor<B, 2>,
}

impl<B> RwkvLmExecutor<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: B::Device,
        model: AutoRegressiveModel<B>,
        max_batch_size: usize,
        num_cells: usize,
        vocab_size: usize,
        embedded_dim: usize,
        num_heads: usize,
        head_size: usize,
    ) -> Self {
        let embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
            .collect();
        let state: Vec<Tensor<B, 4>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, num_heads, head_size, head_size], &device))
            .collect();
        let embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
            .collect();

        let rng_init: Vec<i32> = (0..max_batch_size)
            .map(|i| (i as i32).wrapping_add(1))
            .collect();
        let rng_states =
            Tensor::<B, 1, Int>::from_data(TensorData::new(rng_init, [max_batch_size]), &device);

        // Keep penalties in F32, matching the rapid-sampling reference implementation.
        let penalties = Tensor::<B, 2>::zeros([max_batch_size, vocab_size], (&device, DType::F32));

        Self {
            device,
            model,
            max_batch_size,
            vocab_size,
            embedded_dim,
            num_heads,
            head_size,
            embedded_token_shift_for_time_mix,
            state,
            embedded_token_shift_for_channel_mix,
            rng_states,
            penalties,
        }
    }

    fn make_context_mask(mask_u8: &[u8]) -> Vec<f32> {
        mask_u8
            .iter()
            .copied()
            .map(|m| if m == 0 { 0.0 } else { 1.0 })
            .collect()
    }

    fn reset_penalties_row(&mut self, batch_index: usize) {
        // Keep penalties in F32, matching the rapid-sampling kernel contract (float* penalties).
        let zeros = Tensor::<B, 2>::zeros([1, self.vocab_size], (&self.device, DType::F32));
        self.penalties = self
            .penalties
            .clone()
            .slice_assign([batch_index..batch_index + 1, 0..self.vocab_size], zeros);
    }
}

impl<B> InferExecutor for RwkvLmExecutor<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    fn prefill(&mut self, batch_positions: &[(usize, &[i32], &[u8])]) -> Result<()> {
        if batch_positions.is_empty() {
            return Ok(());
        }

        let context_len = batch_positions[0].1.len();
        if context_len == 0 {
            return Ok(());
        }

        for (_, token_ids, context_mask) in batch_positions {
            if token_ids.len() != context_len || context_mask.len() != context_len {
                return Err(Error::BadRequest(
                    "prefill: inconsistent token_ids/context_mask length".to_string(),
                ));
            }
        }

        let mut flat_tokens = vec![0i32; self.max_batch_size * context_len];
        let mut flat_mask = vec![0f32; self.max_batch_size * context_len];

        for (batch_index, token_ids, context_mask) in batch_positions {
            if *batch_index >= self.max_batch_size {
                return Err(Error::BadRequest(format!(
                    "prefill: batch_index {batch_index} out of range (max_batch_size={})",
                    self.max_batch_size
                )));
            }

            let base = batch_index * context_len;
            flat_tokens[base..base + context_len].copy_from_slice(token_ids);
            let mask_f = Self::make_context_mask(context_mask);
            flat_mask[base..base + context_len].copy_from_slice(&mask_f);
        }

        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(flat_tokens, [self.max_batch_size, context_len]),
            &self.device,
        );
        let context_mask = Tensor::<B, 2>::from_data(
            TensorData::new(flat_mask, [self.max_batch_size, context_len]),
            &self.device,
        );

        let _ = self.model.infer(
            tokens,
            Some(context_mask),
            &mut self.embedded_token_shift_for_time_mix,
            &mut self.state,
            &mut self.embedded_token_shift_for_channel_mix,
            true,
        );
        Ok(())
    }

    fn decode(
        &mut self,
        batch_positions: &[(usize, i32)],
        sampling: SamplingConfig,
    ) -> Result<Vec<(usize, i32)>> {
        if batch_positions.is_empty() {
            return Ok(Vec::new());
        }

        let mut flat_tokens = vec![0i32; self.max_batch_size];
        let mut flat_mask = vec![0f32; self.max_batch_size];
        for (batch_index, token_id) in batch_positions {
            if *batch_index >= self.max_batch_size {
                return Err(Error::BadRequest(format!(
                    "decode: batch_index {batch_index} out of range (max_batch_size={})",
                    self.max_batch_size
                )));
            }
            flat_tokens[*batch_index] = *token_id;
            flat_mask[*batch_index] = 1.0;
        }

        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(flat_tokens, [self.max_batch_size, 1]),
            &self.device,
        );
        let context_mask = Tensor::<B, 2>::from_data(
            TensorData::new(flat_mask, [self.max_batch_size, 1]),
            &self.device,
        );

        let logits = self
            .model
            .infer(
                tokens,
                Some(context_mask),
                &mut self.embedded_token_shift_for_time_mix,
                &mut self.state,
                &mut self.embedded_token_shift_for_channel_mix,
                true,
            )
            .squeeze_dim(1); // [max_batch_size, vocab_size]

        let mut logits_active: Vec<Tensor<B, 2>> = Vec::with_capacity(batch_positions.len());
        let mut rng_active: Vec<Tensor<B, 1, Int>> = Vec::with_capacity(batch_positions.len());
        let mut penalties_active: Vec<Tensor<B, 2>> = Vec::with_capacity(batch_positions.len());

        for (batch_index, _) in batch_positions {
            logits_active.push(
                logits
                    .clone()
                    .slice([*batch_index..*batch_index + 1, 0..self.vocab_size]),
            );
            rng_active.push(
                self.rng_states
                    .clone()
                    .slice([*batch_index..*batch_index + 1]),
            );
            if sampling.penalties_enabled() {
                penalties_active.push(
                    self.penalties
                        .clone()
                        .slice([*batch_index..*batch_index + 1, 0..self.vocab_size]),
                );
            }
        }

        let logits_active = Tensor::cat(logits_active, 0);
        let rng_active = Tensor::cat(rng_active, 0);
        let penalties = if sampling.penalties_enabled() {
            let penalties_active = Tensor::cat(penalties_active, 0);
            Some((
                penalties_active,
                RapidSamplePenaltyConfig {
                    presence_penalty: sampling.presence_penalty,
                    repetition_penalty: sampling.repetition_penalty,
                    penalty_decay: sampling.penalty_decay,
                },
            ))
        } else {
            None
        };

        let out = rapid_sample::<B>(
            logits_active,
            rng_active,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            penalties,
        );

        for (i, (batch_index, _)) in batch_positions.iter().enumerate() {
            let state_i = out.states.clone().slice([i..i + 1]);
            self.rng_states = self
                .rng_states
                .clone()
                .slice_assign([*batch_index..*batch_index + 1], state_i);
        }

        if let Some(updated_penalties) = out.penalties {
            for (i, (batch_index, _)) in batch_positions.iter().enumerate() {
                let penalty_row = updated_penalties
                    .clone()
                    .slice([i..i + 1, 0..self.vocab_size]);
                self.penalties = self.penalties.clone().slice_assign(
                    [*batch_index..*batch_index + 1, 0..self.vocab_size],
                    penalty_row,
                );
            }
        }

        let token_ids = out
            .token_ids
            .to_data()
            .to_vec::<i32>()
            .expect("token_ids to_vec");

        let mut out_pairs = Vec::with_capacity(batch_positions.len());
        for (i, (batch_index, _)) in batch_positions.iter().enumerate() {
            out_pairs.push((*batch_index, token_ids[i]));
        }

        Ok(out_pairs)
    }

    fn reset_batch_position(&mut self, batch_index: usize) -> Result<()> {
        if batch_index >= self.max_batch_size {
            return Err(Error::BadRequest(format!(
                "reset_batch_position: batch_index {batch_index} out of range (max_batch_size={})",
                self.max_batch_size
            )));
        }

        let zeros_shift = Tensor::<B, 2>::zeros([1, self.embedded_dim], &self.device);
        for t in self.embedded_token_shift_for_time_mix.iter_mut() {
            *t = t.clone().slice_assign(
                [batch_index..batch_index + 1, 0..self.embedded_dim],
                zeros_shift.clone(),
            );
        }
        for t in self.embedded_token_shift_for_channel_mix.iter_mut() {
            *t = t.clone().slice_assign(
                [batch_index..batch_index + 1, 0..self.embedded_dim],
                zeros_shift.clone(),
            );
        }

        let zeros_state = Tensor::<B, 4>::zeros(
            [1, self.num_heads, self.head_size, self.head_size],
            &self.device,
        );
        for s in self.state.iter_mut() {
            *s = s.clone().slice_assign(
                [
                    batch_index..batch_index + 1,
                    0..self.num_heads,
                    0..self.head_size,
                    0..self.head_size,
                ],
                zeros_state.clone(),
            );
        }

        let seed = (batch_index as i32).wrapping_add(1);
        let seed_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![seed], [1]), &self.device);
        self.rng_states = self
            .rng_states
            .clone()
            .slice_assign([batch_index..batch_index + 1], seed_tensor);

        self.reset_penalties_row(batch_index);

        Ok(())
    }
}
