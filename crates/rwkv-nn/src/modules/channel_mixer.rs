use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::{Backend, Tensor},
    tensor::activation::relu,
};

use crate::functions::{
    init_weights::{get_token_shift_diff_scale, uniform_init, zeros_init},
    token_shift::token_shift,
};

#[derive(Config, Debug)]

pub struct ChannelMixerConfig {
    num_cells: usize,
    embedded_dim: usize,
}

impl ChannelMixerConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> ChannelMixer<B> {
        ChannelMixer {
            w_in: LinearConfig::new(self.embedded_dim, self.embedded_dim * 4)
                .with_bias(false)
                .init(device),
            w_out: LinearConfig::new(self.embedded_dim * 4, self.embedded_dim)
                .with_bias(false)
                .init(device),
            token_shift_diff_scale: Param::from_tensor(Tensor::empty(
                [1, 1, self.embedded_dim],
                device,
            )),
            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            cell_id,
        }
    }
}

#[derive(Module, Debug)]

pub struct ChannelMixer<B: Backend> {
    pub w_in: Linear<B>,
    pub w_out: Linear<B>,
    pub token_shift_diff_scale: Param<Tensor<B, 3>>,

    pub num_cells: usize,
    pub embedded_dim: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> ChannelMixer<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        let bound = 0.5 / (self.embedded_dim as f64).sqrt();

        uniform_init(&mut self.w_in.weight, -bound, bound);

        zeros_init(&mut self.w_out.weight);

        self.token_shift_diff_scale = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            4.0,
            device,
        ));
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        shift_embedded: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch_size_per_device, context_length, _] = x.dims();

        let x_state_out = x
            .clone()
            .slice([
                0..batch_size_per_device,
                (context_length - 1)..context_length,
            ])
            .squeeze_dim(1);

        let time_shifted_diff = token_shift(x.clone(), shift_embedded) - x.clone();

        let x_in = x.clone() + time_shifted_diff * self.token_shift_diff_scale.val();

        let hidden = self.w_in.forward(x_in);

        let hidden = relu(hidden).powf_scalar(2.0);

        let out = self.w_out.forward(hidden);

        (out, x_state_out)
    }
}
