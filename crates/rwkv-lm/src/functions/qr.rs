use burn::prelude::{Backend, Tensor};

/// Performs QR decomposition on a 2D tensor.
///
/// Implemented using the Modified Gram-Schmidt algorithm for better numerical
/// stability.
///
/// # Arguments
///
/// * `a` - A 2D tensor of shape [M, N].
///
/// # Returns
///
/// A tuple `(Q, R)` where:
/// * `Q` is an orthogonal matrix of shape [M, min(M, N)].
/// * `R` is an upper triangular matrix of shape [min(M, N), N].

pub fn qr<B: Backend>(a: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let device = a.device();

    let [m, n] = a.dims();

    let min_dim = m.min(n);

    let mut q = Tensor::<B, 2>::zeros([m, min_dim], &device);

    let mut r = Tensor::<B, 2>::zeros([min_dim, n], &device);

    let mut v = a;

    for j in 0..min_dim {
        let v_j = v.clone().slice([0..m, j..j + 1]);

        // r_jj = ||v_j||. Use sum_dim to keep dimensions for broadcasting.
        let norm_v_j_tensor = v_j.clone().powf_scalar(2.0).sum_dim(0).sqrt(); // Shape: [1, 1]
        r = r.slice_assign([j..j + 1, j..j + 1], norm_v_j_tensor.clone());

        // q_j = v_j / ||v_j||
        let q_j = v_j / (norm_v_j_tensor + 1e-12);

        q = q.slice_assign([0..m, j..j + 1], q_j.clone());

        // Update remaining vectors
        if j + 1 < n {
            let v_rest = v.clone().slice([0..m, (j + 1)..n]);

            let q_j_t = q_j.clone().transpose();

            // r_jk = q_j^T * v_rest
            let r_jk = q_j_t.matmul(v_rest.clone());

            r = r.slice_assign([j..j + 1, (j + 1)..n], r_jk.clone());

            // v_rest = v_rest - q_j * r_jk
            let v_update = v_rest - q_j.matmul(r_jk);

            v = v.slice_assign([0..m, (j + 1)..n], v_update);
        }
    }

    (q, r)
}
