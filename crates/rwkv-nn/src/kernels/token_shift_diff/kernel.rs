use burn::cubecl;
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn rwkv_token_shift_diff_kernel<F: Float>(
    embedded_context: &Tensor<F>,
    embedded_token_shift: &Tensor<F>,
    context_mask: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let index = ABSOLUTE_POS;
    if index >= output.len() {
        terminate!();
    }

    let context_length = context_mask.shape(1);
    let embedded_dim = embedded_context.shape(2);

    let feature_index = index % embedded_dim;
    let token_index = index / embedded_dim;
    let time_index = token_index % context_length;
    let batch_index = token_index / context_length;

    let mask_index = batch_index * context_length + time_index;
    let mask = context_mask[mask_index];
    let zero = F::new(0.0);

    if mask == zero {
        output[index] = zero;
        terminate!();
    }

    let shift_index = batch_index * embedded_dim + feature_index;
    let prev = if time_index == 0 {
        embedded_token_shift[shift_index]
    } else {
        let prev_mask = context_mask[mask_index - 1];
        if prev_mask == zero {
            embedded_token_shift[shift_index]
        } else {
            embedded_context[index - embedded_dim]
        }
    };

    output[index] = prev - embedded_context[index];
}
