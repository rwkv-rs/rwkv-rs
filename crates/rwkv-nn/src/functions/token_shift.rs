use burn::{Tensor, prelude::Backend};

pub fn token_shift<B: Backend>(x: Tensor<B, 3>, shift_embedded: Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, t, _] = x.dims();

    Tensor::cat(
        vec![
            shift_embedded.unsqueeze_dim(1),
            x.clone().slice([0..b, 0..(t - 1)]),
        ],
        1,
    )
}
