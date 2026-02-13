use burn::prelude::{Backend, Tensor};

/// Inference-time mutable state for `MultiCausalCells`.
///
/// This keeps per-cell tensors in `Vec`s to avoid repeated slicing / slice-assign
/// on large packed tensors during decode.
#[derive(Debug)]
pub struct MultiCausalCellsInferenceState<B: Backend> {
    pub time_mix_state: Vec<Tensor<B, 4>>,   // [batch_size, num_heads, head_size, head_size]
    pub time_mix_shift: Vec<Tensor<B, 2>>,   // [batch_size, embedded_dim]
    pub channel_mix_shift: Vec<Tensor<B, 2>>, // [batch_size, embedded_dim]
}

impl<B: Backend> MultiCausalCellsInferenceState<B> {
    pub fn new_zeros(
        batch_size: usize,
        num_cells: usize,
        embedded_dim: usize,
        num_heads: usize,
        head_size: usize,
        device: &B::Device,
    ) -> Self {
        let time_mix_state = (0..num_cells)
            .map(|_| Tensor::zeros([batch_size, num_heads, head_size, head_size], device))
            .collect();

        let time_mix_shift = (0..num_cells)
            .map(|_| Tensor::zeros([batch_size, embedded_dim], device))
            .collect();

        let channel_mix_shift = (0..num_cells)
            .map(|_| Tensor::zeros([batch_size, embedded_dim], device))
            .collect();

        Self {
            time_mix_state,
            time_mix_shift,
            channel_mix_shift,
        }
    }

    pub fn num_cells(&self) -> usize {
        self.time_mix_state.len()
    }
}

