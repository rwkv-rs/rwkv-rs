use burn::cubecl;
use cubecl::{cube, prelude::*};

#[derive(CubeLaunch, CubeType)]
pub struct TokenShiftDiffInputs<F: Float> {
    pub embedded_context: Tensor<F>,
    pub batch_ids: Tensor<u32>,
    pub context_mask: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct TokenShiftDiffOutputs<F: Float> {
    pub token_shifted_diff: Tensor<F>,
    pub next_token_shift: Tensor<F>,
}

#[cube(launch)]
pub fn rwkv_token_shift_diff_kernel<F: Float>(
    inputs: &TokenShiftDiffInputs<F>,
    outputs: &mut TokenShiftDiffOutputs<F>,
) {
    let index = ABSOLUTE_POS;
    let active_batch_size = inputs.embedded_context.shape(0);
    let embedded_dim = inputs.embedded_context.shape(2);

    if index >= active_batch_size * embedded_dim {
        terminate!();
    }

    let context_length = inputs.context_mask.shape(1);
    let feature_index = index % embedded_dim;
    let batch_index = index / embedded_dim;
    let state_index = inputs.batch_ids[batch_index] as usize;
    let shift_index = state_index * embedded_dim + feature_index;

    // This kernel updates the full `[max_batch, C]` shift buffer in place. We therefore bind one
    // thread to one `(active_batch, feature)` lane and iterate over time locally so the old shift
    // value is read exactly once before the final writeback. A flat `(batch, time, feature)` kernel
    // would race against its own in-place update.
    let external_shift = outputs.next_token_shift[shift_index];
    let zero = F::new(0.0);

    let mut prev = external_shift;
    let mut next_shift = external_shift;

    for time_index in 0..context_length {
        let mask_index = batch_index * context_length + time_index;
        let output_index = (batch_index * context_length + time_index) * embedded_dim + feature_index;
        let mask = inputs.context_mask[mask_index];

        if mask == zero {
            outputs.token_shifted_diff[output_index] = zero;
            prev = external_shift;
        } else {
            let current = inputs.embedded_context[output_index];
            outputs.token_shifted_diff[output_index] = prev - current;
            prev = current;
            next_shift = current;
        }
    }

    outputs.next_token_shift[shift_index] = next_shift;
}
