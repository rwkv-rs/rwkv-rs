use burn::{
    module::Param,
    prelude::{Backend, Tensor},
    tensor::Distribution,
};

use crate::functions::qr::qr;

pub fn uniform_init<B: Backend>(param: &mut Param<Tensor<B, 2>>, low: f64, high: f64) {
    let shape = param.shape();

    let device = &param.device();

    let uniform_tensor = Tensor::random(shape, Distribution::Uniform(low, high), device);

    *param = Param::from_tensor(uniform_tensor);
}

pub fn zeros_init<B: Backend, const D: usize>(param: &mut Param<Tensor<B, D>>) {
    let shape = param.shape();

    let device = &param.device();

    let zeros_tensor = Tensor::zeros(shape, device);

    *param = Param::from_tensor(zeros_tensor);
}

pub fn ones_init<B: Backend, const D: usize>(param: &mut Param<Tensor<B, D>>) {
    let shape = param.shape();

    let device = &param.device();

    let ones_tensor = Tensor::ones(shape, device);

    *param = Param::from_tensor(ones_tensor);
}

pub fn constant_init<B: Backend, const D: usize>(param: &mut Param<Tensor<B, D>>, value: f64) {
    let shape = param.shape();

    let device = &param.device();

    let constant_tensor = Tensor::ones(shape, device) * value;

    *param = Param::from_tensor(constant_tensor);
}

pub fn calculate_decay_speed<B: Backend>(
    num_cells: usize,
    embedded_dim: usize,
    head_size: usize,

    cell_id: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let decay_base_vec = get_decay_base_vec(num_cells, embedded_dim, cell_id);

    let zigzag_vec = get_zigzag_vec(embedded_dim, head_size);

    let combined_vec: Vec<f64> = decay_base_vec
        .iter()
        .zip(zigzag_vec.iter())
        .map(|(w, z)| w + 0.5 + z * 2.5)
        .collect();

    Tensor::from_floats(combined_vec.as_slice(), device)
}

pub fn get_token_shift_diff_scale<B: Backend>(
    num_cells: usize,
    embedded_dim: usize,
    cell_id: usize,
    power: f64,
    device: &B::Device,
) -> Tensor<B, 3> {
    let layer_ratio = 1.0 - (cell_id as f64 / num_cells as f64);

    let ratio_tensor = get_dim_ratio_tensor::<B>(embedded_dim, device);

    let one_tensor = Tensor::ones([1, 1, embedded_dim], device);

    one_tensor - ratio_tensor.powf_scalar(power * layer_ratio)
}

pub fn calculate_token_shift_with_offset<B: Backend>(
    num_cells: usize,
    embedded_dim: usize,
    cell_id: usize,
    power: f64,
    device: &B::Device,
) -> Tensor<B, 3> {
    let layer_ratio = 1.0 - (cell_id as f64 / num_cells as f64);

    let ratio_tensor = get_dim_ratio_tensor::<B>(embedded_dim, device);

    let one_tensor = Tensor::ones([1, 1, embedded_dim], device);

    // The reference implementation for x_k and x_v does not use an offset.
    one_tensor - ratio_tensor.powf_scalar(power * layer_ratio)
}

pub fn orthogonal_init<B: Backend>(param: &mut Param<Tensor<B, 2>>, gain: Option<f32>) {
    let [rows, cols] = param.dims();

    let device = &param.device();

    // 1. Create a random matrix.
    let random_tensor = Tensor::random([rows, cols], Distribution::Normal(0.0, 1.0), device);

    // 2. Perform QR decomposition. PyTorch's implementation handles non-square
    //    matrices
    // by performing QR on the transposed matrix if rows < cols.
    let (q_interim, r) = if rows >= cols {
        qr(random_tensor)
    } else {
        let (q_t, r_t) = qr(random_tensor.transpose());

        (q_t.transpose(), r_t)
    };

    // 3. Manually extract the diagonal of R to determine the signs.
    let min_dim = r.dims()[0].min(r.dims()[1]);

    let mut diag_elems = Vec::with_capacity(min_dim);

    for i in 0..min_dim {
        diag_elems.push(r.clone().slice([i..i + 1, i..i + 1]));
    }

    let r_diag: Tensor<B, 1> = Tensor::cat(diag_elems, 1).squeeze_dim(0);

    let signs = r_diag.sign();

    // 4. Ensure Q has the correct final dimensions and apply signs for determinism.
    let q = q_interim.slice([0..rows, 0..cols]);

    let q_signed = if rows >= cols {
        // Normal case: multiply each column by corresponding sign
        q * signs.unsqueeze_dim(0) // [rows, cols] * [1, cols]
    } else {
        // Transposed case: multiply each row by corresponding sign
        q * signs.unsqueeze_dim(1) // [rows, cols] * [rows, 1]
    };

    // 5. Apply gain.
    let result = if let Some(g) = gain {
        q_signed * g
    } else {
        q_signed
    };

    *param = Param::from_tensor(result);
}

fn get_dim_ratio_tensor<B: Backend>(embedded_dim: usize, device: &B::Device) -> Tensor<B, 3> {
    let ratio_vec: Vec<f64> = (0..embedded_dim)
        .map(|i| i as f64 / embedded_dim as f64)
        .collect();

    let ratio_tensor: Tensor<B, 1> = Tensor::from_floats(ratio_vec.as_slice(), device);

    ratio_tensor.reshape([1, 1, embedded_dim])
}

fn get_centered_linear_vec(dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|n| n as f64 / (dim - 1).max(1) as f64 - 0.5)
        .collect()
}

fn get_zigzag_vec(embedded_dim: usize, head_size: usize) -> Vec<f64> {
    let half_head_size = (head_size as f64 - 1.0) / 2.0;

    (0..embedded_dim)
        .map(|n| {
            let val = ((n as f64 % head_size as f64) - half_head_size) / half_head_size.max(1.0);

            val * val.abs()
        })
        .collect()
}

fn get_decay_base_vec(num_cells: usize, embedded_dim: usize, cell_id: usize) -> Vec<f64> {
    let layer_ratio = if num_cells > 1 {
        cell_id as f64 / (num_cells - 1) as f64
    } else {
        0.0
    };

    (0..embedded_dim)
        .map(|n| {
            let dim_ratio = n as f64 / (embedded_dim - 1).max(1) as f64;

            let power = 1.0 + layer_ratio.powf(0.3);

            -6.0 + 6.0 * dim_ratio.powf(power)
        })
        .collect()
}

pub fn get_learning_rate_lora_bias<B: Backend>(
    embedded_dim: usize,
    head_size: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let linear_vec = get_centered_linear_vec(embedded_dim);

    let zigzag_vec = get_zigzag_vec(embedded_dim, head_size);

    let combined_vec: Vec<f64> = linear_vec
        .iter()
        .zip(zigzag_vec.iter())
        .map(|(l, z)| -0.19 + z * 0.3 + l * 0.4)
        .collect();

    Tensor::from_floats(combined_vec.as_slice(), device)
}

pub fn get_value_lora_bias<B: Backend>(embedded_dim: usize, device: &B::Device) -> Tensor<B, 1> {
    let linear_vec = get_centered_linear_vec(embedded_dim);

    let combined_vec: Vec<f64> = linear_vec.iter().map(|l| 0.73 - l * 0.4).collect();

    Tensor::from_floats(combined_vec.as_slice(), device)
}
