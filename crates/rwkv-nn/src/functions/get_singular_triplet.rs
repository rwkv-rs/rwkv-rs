use burn::{
    prelude::{Backend, Tensor},
    tensor::{Distribution, cast::ToElement},
};

use crate::functions::normalize::normalize;

pub fn get_singular_triplet<B: Backend, const D: usize>(
    w: Tensor<B, D>,
    u_state: Option<Tensor<B, D>>,
    iters: usize,
) -> (f32, Tensor<B, D>, Tensor<B, D>) {
    assert_eq!(D, 2, "SVD function only supports 2D tensors");

    let m = w.dims()[0];

    let n = w.dims()[1];

    let u0 = u_state.unwrap_or_else(|| {
        Tensor::<B, D>::random([m, 1], Distribution::Normal(0.0, 1.0), &w.device())
    });

    let mut u = normalize(u0, 2.0, 0, 1e-12);

    let mut v = Tensor::<B, D>::random([n, 1], Distribution::Normal(0.0, 1.0), &w.device());

    v = normalize(v, 2.0, 0, 1e-12);

    for _ in 0..iters {
        v = normalize(w.clone().transpose().matmul(u.clone()), 2.0, 0, 1e-12);

        u = normalize(w.clone().matmul(v.clone()), 2.0, 0, 1e-12);
    }

    let wv = w.clone().matmul(v.clone());

    let sigma = u.clone().transpose().matmul(wv).into_scalar().to_f32();

    (sigma, u, v)
}
