use burn::prelude::{Backend, Tensor};

pub fn lerp<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    end: Tensor<B, D>,
    weight: Tensor<B, D>,
) -> Tensor<B, D> {
    input.clone() + (end - input) * weight.clone()
}
