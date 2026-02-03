use burn::{
    config::Config,
    module::{Module, Param},
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::{
    functions::init_weights::{constant_init, get_token_shift_diff_scale, zeros_init},
    layers::lora::{ActivationFn, LoRA, LoRAConfig, LoRAType},
};

#[derive(Config, Debug)]
pub struct GatedReadoutConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    output_gate_lora_rank: usize,
}

impl GatedReadoutConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> GatedReadout<B> {
        let empty_param = Param::from_tensor(Tensor::empty([1, 1, self.embedded_dim], device));

        let projection_output = LinearConfig::new(self.embedded_dim, self.embedded_dim)
            .with_bias(false)
            .init(device);

        GatedReadout {
            param_gate: empty_param,

            param_output_gate_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.output_gate_lora_rank,
                self.head_size,
                false,
                ActivationFn::Sigmoid,
            )
            .init(device),

            param_receptance_key_bonus: Param::from_tensor(Tensor::empty(
                [self.num_heads, self.embedded_dim / self.num_heads],
                device,
            )),
            group_norm: GroupNormConfig::new(self.num_heads, self.embedded_dim)
                .with_epsilon(64e-5)
                .init(device),
            projection_output,

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,

            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct GatedReadout<B: Backend> {
    param_gate: Param<Tensor<B, 3, Float>>,

    param_output_gate_lora: LoRA<B>,

    param_receptance_key_bonus: Param<Tensor<B, 2>>,
    group_norm: GroupNorm<B>,
    projection_output: Linear<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> GatedReadout<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.param_gate = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.2,
            device,
        ));

        self.param_output_gate_lora
            .init_weight(self.cell_id, LoRAType::LearningRate, device);

        constant_init(&mut self.param_receptance_key_bonus, -0.04);

        zeros_init(&mut self.projection_output.weight);

        if let Some(ref mut gamma) = self.group_norm.gamma {
            let layer_scale = ((1 + self.cell_id) as f64 / self.num_cells as f64).powf(0.7);

            constant_init(gamma, layer_scale);
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        time_shifted_diff: Tensor<B, 3>,
        wkv_output: Tensor<B, 4>,
        wkv_receptance_input: Tensor<B, 4>,
        wkv_key_input: Tensor<B, 4>,
        wkv_value_input: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch_size_per_device, context_length, embedded_dim] = x.dims();

        let x_gate = x + time_shifted_diff * self.param_gate.val();

        let gate = self.param_output_gate_lora.forward(x_gate);

        let out = wkv_output.reshape([batch_size_per_device, context_length, embedded_dim]);

        let out: Tensor<B, 2> = out.reshape([batch_size_per_device * context_length, embedded_dim]);

        let out_normalized = self.group_norm.forward(out).reshape([
            batch_size_per_device,
            context_length,
            embedded_dim,
        ]);

        let bonus: Tensor<B, 4> = (wkv_receptance_input
            * wkv_key_input
            * self
                .param_receptance_key_bonus
                .val()
                .unsqueeze_dims(&[0, 1]))
        .sum_dim(3)
            * wkv_value_input;

        let bonus: Tensor<B, 3> = bonus.reshape([batch_size_per_device, context_length, embedded_dim]);

        let out_gated = (out_normalized + bonus) * gate;

        self.projection_output.forward(out_gated)
    }
}
