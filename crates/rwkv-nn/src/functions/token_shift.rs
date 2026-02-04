use burn::{Tensor, prelude::Backend};

pub fn token_shift<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch_size, context_length, _] = embedded_context.dims();

    Tensor::cat(
        vec![
            embedded_token_shift.unsqueeze_dim(1),
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
