use burn::{prelude::Backend, Tensor};

/// Apply `context_mask` to `embedded_context`, forcing masked timesteps to be strict zeros.
///
/// - `embedded_context`: [batch_size, context_length, embedded_dim]
/// - `context_mask`: [batch_size, context_length] with values 0/1
pub fn apply_context_mask<B: Backend>(
    embedded_context: Tensor<B, 3>,
    context_mask: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch_size, context_length, _embedded_dim] = embedded_context.dims();
    let [mask_batch_size, mask_context_length] = context_mask.dims();

    assert_eq!(
        (batch_size, context_length),
        (mask_batch_size, mask_context_length),
        "context_mask shape mismatch with embedded_context"
    );

    embedded_context * context_mask.unsqueeze_dim(2)
}

/// Get the last timestep mask.
///
/// - input: [batch_size, context_length]
/// - output: [batch_size, 1]
pub fn get_context_mask_last<B: Backend>(context_mask: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, context_length] = context_mask.dims();
    context_mask.slice([0..batch_size, (context_length - 1)..context_length])
}

/// Count valid tokens per batch lane.
///
/// - input: [batch_size, context_length] with values 0/1
/// - output: [batch_size]
pub fn num_tokens_valid<B: Backend>(context_mask: Tensor<B, 2>) -> Tensor<B, 1> {
    context_mask.sum_dim(1).squeeze_dim(1)
}

