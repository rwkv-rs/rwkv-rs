mod gated_readout;
pub mod param_state;
mod weight_prepare;

use burn::{config::Config, module::Module, prelude::*};

use crate::{
    kernels::wkv7_common::Wkv7Kernel,
    layers::lora::LoRARanks,
};

use crate::functions::token_shift::get_embedded_token_shift;
use crate::functions::context_mask::apply_context_mask;
use crate::modules::time_mixer::gated_readout::GatedReadoutInput;
use gated_readout::{GatedReadout, GatedReadoutConfig};
use weight_prepare::{WeightPrepare, WeightPrepareConfig};

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

    pub fn forward<K: Wkv7Kernel<B>>(&self, time_mixer_input: TimeMixerIO<B>) -> TimeMixerIO<B> {
        let TimeMixerIO {
            embedded_context,
            context_mask,
            value_from_first_cell,
            embedded_token_shift,
            state,
        } = time_mixer_input;

        let [batch_size_per_device, context_length, _embedded_dim] = embedded_context.dims();

        let (num_heads, head_size) = (self.num_heads, self.head_size);
        
        let embedded_context = apply_context_mask(embedded_context, context_mask.clone());
        let value_from_first_cell = apply_context_mask(value_from_first_cell, context_mask.clone());

        let output_embedded_token_shift = match embedded_token_shift {
            Some(_) => Some(get_embedded_token_shift(embedded_context.clone())),
            None => None,
        };

        let weight_prepare_output = self.weight_prepare.forward(
            embedded_context.clone(),
            value_from_first_cell.clone(),
            embedded_token_shift.clone(),
            context_mask.clone(),
        );

        let shape = [batch_size_per_device, context_length, num_heads, head_size];
        let wkv7_forward_input = weight_prepare_output.reshape_to_wkv7_input(shape);
        let wkv7_forward_output = K::forward(wkv7_forward_input.clone(), state, 16);
        let gated_readout_input = GatedReadoutInput {
            embedded_context,
            token_shifted_diff: weight_prepare_output.token_shifted_diff,
            wkv7_forward_output: wkv7_forward_output.output,
            wkv7_forward_input_receptance: wkv7_forward_input.receptance,
            wkv7_forward_input_replacement_key: wkv7_forward_input.replacement_key,
            wkv7_forward_input_value: wkv7_forward_input.value,
        };

        let output_embedded_context = self.gated_readout.forward(gated_readout_input);

        TimeMixerIO {
            embedded_context: apply_context_mask(
                output_embedded_context,
                context_mask.clone()
            ),
            context_mask: context_mask.clone(),
            value_from_first_cell: apply_context_mask(
                weight_prepare_output.value_from_first_cell,
                context_mask,
            ),
            embedded_token_shift: output_embedded_token_shift,
            state: wkv7_forward_output.next_state,
        }
    }
}

pub struct TimeMixerIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>,
    pub context_mask: Option<Tensor<B, 2>>,
    pub value_from_first_cell: Tensor<B, 3>,
    pub embedded_token_shift: Option<Tensor<B, 2>>,
    pub state: Option<Tensor<B, 4>>,
}
