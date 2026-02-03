mod gated_readout;
mod weight_prepare;

use burn::{
    config::Config,
    module::Module,
    prelude::*,
};

use crate::{
    kernels::wkv7_pretrain::{wkv7_pretrain_forward, Wkv7PretrainBackend},
    layers::lora::LoRARanks,
};

use gated_readout::{GatedReadout, GatedReadoutConfig};
use weight_prepare::{WeightPrepare, WeightPrepareConfig, WeightPrepareOutput};

#[derive(Config, Debug)]
pub struct TimeMixerConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl TimeMixerConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> TimeMixer<B> {
        let lora_ranks_by_dim = [
            LoRARanks {
                min_d_model: 0,
                weight_decay_lora: 64,
                learning_rate_lora: 64,
                value_residual_lora: 32,
                output_gate_lora: 128,
            },
            LoRARanks {
                min_d_model: 2048,
                weight_decay_lora: 128,
                learning_rate_lora: 64,
                value_residual_lora: 64,
                output_gate_lora: 256,
            },
            LoRARanks {
                min_d_model: 4096,
                weight_decay_lora: 192,
                learning_rate_lora: 96,
                value_residual_lora: 96,
                output_gate_lora: 384,
            },
            LoRARanks {
                min_d_model: 6144,
                weight_decay_lora: 256,
                learning_rate_lora: 128,
                value_residual_lora: 128,
                output_gate_lora: 512,
            },
        ];

        let lora_rank = lora_ranks_by_dim
            .iter()
            .rev()
            .find(|rank| rank.min_d_model <= self.embedded_dim)
            .unwrap();

        TimeMixer {
            weight_prepare: WeightPrepareConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
                lora_rank.weight_decay_lora,
                lora_rank.learning_rate_lora,
                lora_rank.value_residual_lora,
            )
            .init(cell_id, device),
            gated_readout: GatedReadoutConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
                lora_rank.output_gate_lora,
            )
            .init(cell_id, device),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,

            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct TimeMixer<B: Backend> {
    weight_prepare: WeightPrepare<B>,
    gated_readout: GatedReadout<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> TimeMixer<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.weight_prepare.init_weights(device);
        self.gated_readout.init_weights(device);
    }

    pub fn forward(
        &self,
        embedded_context: Tensor<B, 3>,
        value_first_layer: Tensor<B, 3>,
        embedded_token_shift: Tensor<B, 2>,
        state: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 4>)
    where
        B: Wkv7PretrainBackend,
    {
        let [batch_size_per_device, context_length, _embedded_dim] = embedded_context.dims();

        let (num_heads, head_size) = (self.num_heads, self.head_size);

        let output_embedded_token_shift = embedded_context
            .clone()
            .slice([
                0..batch_size_per_device,
                (context_length - 1)..context_length,
            ])
            .squeeze_dim(1);

        let WeightPrepareOutput {
            token_shifted_diff,
            value_first_layer,
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
        } = self
            .weight_prepare
            .forward(embedded_context.clone(), value_first_layer.clone(), embedded_token_shift.clone());

        let wkv_receptance_input: Tensor<B, 4> =
            receptance.reshape([batch_size_per_device, context_length, num_heads, head_size]);

        let wkv_weight_decay_input: Tensor<B, 4> =
            weight_decay.reshape([batch_size_per_device, context_length, num_heads, head_size]);

        let wkv_key_input: Tensor<B, 4> =
            replacement_key.reshape([batch_size_per_device, context_length, num_heads, head_size]);

        let wkv_value_input: Tensor<B, 4> =
            value.reshape([batch_size_per_device, context_length, num_heads, head_size]);

        let wkv_removal_input: Tensor<B, 4> = removal_key_normalized.reshape([
            batch_size_per_device,
            context_length,
            num_heads,
            head_size,
        ]);

        let wkv_replacement_input: Tensor<B, 4> =
            replacement.reshape([batch_size_per_device, context_length, num_heads, head_size]);

        let wkv_out = wkv7_pretrain_forward(
            wkv_weight_decay_input.clone(),
            wkv_receptance_input.clone(),
            wkv_key_input.clone(),
            wkv_value_input.clone(),
            wkv_removal_input.clone(),
            wkv_replacement_input.clone(),
            16,
        );

        let _current_vk_state = state.clone();

        let output_embedded_context = self.gated_readout.forward(
            embedded_context,
            token_shifted_diff,
            wkv_out.output,
            wkv_receptance_input,
            wkv_key_input,
            wkv_value_input,
        );

        (output_embedded_context, value_first_layer, output_embedded_token_shift, state)
    }
}
