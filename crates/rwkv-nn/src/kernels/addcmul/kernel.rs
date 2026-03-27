use burn::cubecl;
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn addcmul_kernel<F: Float>(
    base: &Tensor<F>,
    diff: &Tensor<F>,
    scale: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let index = ABSOLUTE_POS;
    if index >= output.len() {
        terminate!();
    }

    let embedded_dim = scale.len();
    let channel_index = index % embedded_dim;

    output[index] = base[index] + diff[index] * scale[channel_index];
}

#[cube(launch)]
pub fn addcmul5_kernel<F: Float>(
    base: &Tensor<F>,
    diff: &Tensor<F>,
    receptance_scale: &Tensor<F>,
    weight_decay_scale: &Tensor<F>,
    key_scale: &Tensor<F>,
    value_scale: &Tensor<F>,
    learning_rate_scale: &Tensor<F>,
    receptance_output: &mut Tensor<F>,
    weight_decay_output: &mut Tensor<F>,
    key_output: &mut Tensor<F>,
    value_output: &mut Tensor<F>,
    learning_rate_output: &mut Tensor<F>,
) {
    let index = ABSOLUTE_POS;
    if index >= base.len() {
        terminate!();
    }

    let embedded_dim = receptance_scale.len();
    let channel_index = index % embedded_dim;

    let current = base[index];
    let delta = diff[index];

    receptance_output[index] = current + delta * receptance_scale[channel_index];
    weight_decay_output[index] = current + delta * weight_decay_scale[channel_index];
    key_output[index] = current + delta * key_scale[channel_index];
    value_output[index] = current + delta * value_scale[channel_index];
    learning_rate_output[index] = current + delta * learning_rate_scale[channel_index];
}
