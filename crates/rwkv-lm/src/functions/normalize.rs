use burn::prelude::{Backend, Tensor};

pub fn normalize<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    p: f32,
    dim: isize,
    epsilon: f32,
) -> Tensor<B, D> {
    let adjusted_dim = if dim < 0 {
        let dim = dim + D as isize;

        assert!(dim >= 0, "Dimension out of range (adjusted_dim={})", dim);

        dim as usize
    } else {
        dim as usize
    };

    assert!(
        adjusted_dim < D,
        "Dimension out of range (input has {} dimensions, but got dim={})",
        D,
        dim
    );

    let norm = input
        .clone()
        .abs()
        .powf_scalar(p)
        .sum_dim(adjusted_dim)
        .powf_scalar(1.0 / p);

    let denom = norm.clamp_min(epsilon);

    input.div(denom)
}
