use rwkv::custom::Tensor;
use rwkv::custom::prelude::{Backend, Int, TensorData};
use rwkv::data::tokenizer::Tokenizer;
use rwkv::nn::kernels::rapid_sample::{RapidSampleBackend, rapid_sample};
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;
use rwkv_infer::engine::InferExecutor;
use rwkv_infer::{Error, Result, SamplingConfig};

use crate::model::AutoRegressiveModel;

rwkv::custom_mode!();

pub struct RwkvLmExecutor<B: Backend>
where
    B: Wkv7Backend + RapidSampleBackend,
{
    device: B::Device,
    tokenizer: Tokenizer,
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
}

impl<B> RwkvLmExecutor<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: B::Device,
        tokenizer_vocab_path: &str,
        model: AutoRegressiveModel<B>,
        max_batch_size: usize,
        num_cells: usize,
        vocab_size: usize,
        embedded_dim: usize,
        num_heads: usize,
        head_size: usize,
    ) -> Self {
        let tokenizer = Tokenizer::new(tokenizer_vocab_path).expect("load tokenizer vocab");

        let embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
            .collect();
        let state: Vec<Tensor<B, 4>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, num_heads, head_size, head_size], &device))
            .collect();
        let embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
            .collect();

        // Deterministic initial RNG state per batch position.
        let rng_init: Vec<i32> = (0..max_batch_size)
            .map(|i| (i as i32).wrapping_add(1))
            .collect();
        let rng_states =
            Tensor::<B, 1, Int>::from_data(TensorData::new(rng_init, [max_batch_size]), &device);

        Self {
            device,
            tokenizer,
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
        }
    }

    fn make_context_mask(&self, mask_u8: &[u8]) -> Vec<f32> {
        mask_u8
            .iter()
            .copied()
            .map(|m| if m == 0 { 0.0 } else { 1.0 })
            .collect()
    }
}

impl<B> InferExecutor for RwkvLmExecutor<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .into_iter()
            .map(|t| t as i32)
            .collect())
    }

    fn detokenize(&self, token_ids: &[i32]) -> Result<String> {
        Ok(self.tokenizer.decode(
            token_ids
                .iter()
                .copied()
                .map(|t| t as u16)
                .collect::<Vec<_>>(),
        ))
    }

    fn prefill(&mut self, batch_positions: &[(usize, &[i32], &[u8])]) -> Result<()> {
        if batch_positions.is_empty() {
            return Ok(());
        }

        let context_length = batch_positions[0].1.len();
        if context_length == 0 {
            return Ok(());
        }

        for (_, tokens, mask) in batch_positions {
            if tokens.len() != context_length || mask.len() != context_length {
                return Err(Error::BadRequest(
                    "prefill: inconsistent tokens/mask length".to_string(),
                ));
            }
        }

        let mut flat_tokens = vec![0i32; self.max_batch_size * context_length];
        let mut flat_mask = vec![0f32; self.max_batch_size * context_length];

        for (batch_index, tokens, mask) in batch_positions {
            if *batch_index >= self.max_batch_size {
                return Err(Error::BadRequest(format!(
                    "prefill: batch_index {batch_index} out of range (max_batch_size={})",
                    self.max_batch_size
                )));
            }
            let base = batch_index * context_length;
            flat_tokens[base..base + context_length].copy_from_slice(tokens);
            let mask_f = self.make_context_mask(mask);
            flat_mask[base..base + context_length].copy_from_slice(&mask_f);
        }

        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(flat_tokens, [self.max_batch_size, context_length]),
            &self.device,
        );
        let context_mask = Tensor::<B, 2>::from_data(
            TensorData::new(flat_mask, [self.max_batch_size, context_length]),
            &self.device,
        );

        // `need_full_logits=true` is the fast path: only the last-token logits are computed.
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
        }

        let logits_active = Tensor::cat(logits_active, 0);
        let rng_active = Tensor::cat(rng_active, 0);

        let out = rapid_sample::<B>(
            logits_active,
            rng_active,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            None,
        );

        for (i, (batch_index, _)) in batch_positions.iter().enumerate() {
            let state_i = out.states.clone().slice([i..i + 1]);
            self.rng_states = self
                .rng_states
                .clone()
                .slice_assign([*batch_index..*batch_index + 1], state_i);
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

        Ok(())
    }
}
