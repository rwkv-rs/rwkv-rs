use burn::{Tensor, prelude::Backend};

pub fn token_shift<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();

    Tensor::cat(
        vec![
            embedded_token_shift.unwrap_or(
                Tensor::zeros([batch_size, embedded_dim], &embedded_context.device()),
            ).unsqueeze_dim(1),
            embedded_context
                .clone()
                .slice([0..batch_size, 0..(context_length - 1)]),
        ],
        1,
    )
}
 
pub fn get_embedded_token_shift<B: Backend>(embedded_context: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, context_length, _] = embedded_context.dims();
    embedded_context.clone()
        .slice([0..batch_size, (context_length - 1)..context_length])
        .squeeze_dim(1)
}

/// Compute token-shifted diff for inference with left-padding support.
///
/// This assumes `context_mask` is used for left padding inside the current chunk, i.e. each
/// batch lane mask is expected to be a prefix of 0s followed by 1s.
///
/// IMPORTANT: `embedded_context` must already be masked so that padded timesteps are strict zeros
/// (e.g. via `apply_context_mask`). This function only fixes the *shift semantics* across left
/// padding; it does not zero the inputs itself.
///
/// - `embedded_context`: [batch_size, context_length, embedded_dim]
/// - `embedded_token_shift`: [batch_size, embedded_dim] (previous token from the last step / chunk)
/// - `context_mask`: [batch_size, context_length] with values 0/1
///
/// Returns `token_shift(embedded_context, embedded_token_shift) - embedded_context`, but with the
/// crucial difference that for the first valid token after left-padding, the "previous token"
/// remains `embedded_token_shift` (padding timesteps do not shift the previous token forward).
pub fn token_shifted_diff_with_context_mask<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Tensor<B, 2>,
    context_mask: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();
    let [mask_batch_size, mask_context_length] = context_mask.dims();

    assert_eq!(
        (batch_size, context_length),
        (mask_batch_size, mask_context_length),
        "context_mask shape mismatch with embedded_context"
    );
    assert_eq!(
        embedded_token_shift.dims(),
        [batch_size, embedded_dim],
        "embedded_token_shift shape mismatch with embedded_context"
    );

    // Standard previous-token shift: [shift, x[:, 0..T-1]].
    let shifted = Tensor::cat(
        vec![
            embedded_token_shift.clone().unsqueeze_dim(1),
            embedded_context
                .clone()
                .slice([0..batch_size, 0..(context_length - 1)]),
        ],
        1,
    );

    // `prev_valid_mask[t] = context_mask[t-1]` (t=0 treated as 0).
    let prev_valid_mask = Tensor::cat(
        vec![
            Tensor::zeros([batch_size, 1], &embedded_context.device()),
            context_mask
                .clone()
                .slice([0..batch_size, 0..(context_length - 1)]),
        ],
        1,
    );

    // If the previous timestep is masked (padding), keep using the external shift state.
    // This handles left padding where the prefix padding should not advance the previous token.
    let use_external_shift = Tensor::ones([batch_size, context_length], &embedded_context.device())
        - prev_valid_mask;

    let external_shift = embedded_token_shift.unsqueeze_dim(1);
    let prev_token =
        shifted.clone() + use_external_shift.unsqueeze_dim(2) * (external_shift - shifted);

    // Only keep diffs on valid tokens.
    (prev_token - embedded_context) * context_mask.unsqueeze_dim(2)
}
