use burn::{
    module::{Module, Param},
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::{sigmoid, softplus},
};

use crate::{
    functions::{
        init_weights::{
            calculate_token_shift_with_offset, constant_init, get_token_shift_diff_scale,
            uniform_init, zeros_init,
        },
        lerp::lerp,
        normalize::normalize,
        token_shift::token_shift,
    },
    kernels::wkv7_pretrain::{Wkv7PretrainBackend, wkv7_pretrain_forward},
    layers::lora::{ActivationFn, LoRA, LoRAConfig, LoRARanks, LoRAType},
};

type TimeMixerWeightPrepareOutput<B> = (
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
);

#[derive(Config, Debug)]
pub struct TimeMixerConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl TimeMixerConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> TimeMixer<B> {
        let empty_param = Param::from_tensor(Tensor::empty([1, 1, self.embedded_dim], device));

        let projection = LinearConfig::new(self.embedded_dim, self.embedded_dim)
            .with_bias(false)
            .init(device);

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
            param_receptance: empty_param.clone(),
            param_weight_decay: empty_param.clone(),
            param_key: empty_param.clone(),
            param_value: empty_param.clone(),
            param_learning_rate: empty_param.clone(),
            param_gate: empty_param.clone(),

            projection_receptance: projection.clone(),
            projection_key: projection.clone(),
            projection_value: projection.clone(),
            projection_output: projection.clone(),

            param_weight_decay_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                lora_rank.weight_decay_lora,
                self.head_size,
                true,
                ActivationFn::Tanh,
            )
            .init(device),
            param_learning_rate_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                lora_rank.learning_rate_lora,
                self.head_size,
                true,
                ActivationFn::NoOP,
            )
            .init(device),
            param_output_gate_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                lora_rank.output_gate_lora,
                self.head_size,
                false,
                ActivationFn::Sigmoid,
            )
            .init(device),
            param_value_residual_lora: if cell_id > 0 {
                Some(
                    LoRAConfig::new(
                        self.num_cells,
                        self.embedded_dim,
                        lora_rank.value_residual_lora,
                        self.head_size,
                        true,
                        ActivationFn::NoOP,
                    )
                    .init(device),
                )
            } else {
                None
            },
            param_key_removal: empty_param.clone(),
            param_key_replacement: empty_param.clone(),
            param_receptance_key_bonus: Param::from_tensor(Tensor::empty(
                [self.num_heads, self.embedded_dim / self.num_heads],
                device,
            )),
            group_norm: GroupNormConfig::new(self.num_heads, self.embedded_dim)
                .with_epsilon(64e-5)
                .init(device),

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
    param_receptance: Param<Tensor<B, 3, Float>>,
    param_weight_decay: Param<Tensor<B, 3, Float>>,
    param_key: Param<Tensor<B, 3, Float>>,
    param_value: Param<Tensor<B, 3, Float>>,
    param_learning_rate: Param<Tensor<B, 3, Float>>,
    param_gate: Param<Tensor<B, 3, Float>>,

    projection_receptance: Linear<B>,
    projection_key: Linear<B>,
    projection_value: Linear<B>,
    projection_output: Linear<B>,

    param_weight_decay_lora: LoRA<B>,
    param_learning_rate_lora: LoRA<B>,
    param_output_gate_lora: LoRA<B>,
    param_value_residual_lora: Option<LoRA<B>>,

    param_key_removal: Param<Tensor<B, 3>>,
    param_key_replacement: Param<Tensor<B, 3>>,

    param_receptance_key_bonus: Param<Tensor<B, 2>>,
    group_norm: GroupNorm<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
    cell_id: usize,
}

impl<B: Backend> TimeMixer<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.param_receptance = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.2,
            device,
        ));

        self.param_weight_decay = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.9,
            device,
        ));

        self.param_key = Param::from_tensor(calculate_token_shift_with_offset(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.7,
            device,
        ));

        self.param_value = Param::from_tensor(calculate_token_shift_with_offset(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.7,
            device,
        ));

        self.param_learning_rate = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.9,
            device,
        ));

        self.param_gate = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.2,
            device,
        ));

        let embedded_dim = self.embedded_dim as f64;

        let receptance_bound = 0.5 / embedded_dim.sqrt();

        let key_bound = 0.05 / embedded_dim.sqrt();

        let value_bound = 0.5 / embedded_dim.sqrt();

        uniform_init(
            &mut self.projection_receptance.weight,
            -receptance_bound,
            receptance_bound,
        );

        uniform_init(&mut self.projection_key.weight, -key_bound, key_bound);
        uniform_init(&mut self.projection_value.weight, -value_bound, value_bound);
        zeros_init(&mut self.projection_output.weight);

        self.param_weight_decay_lora
            .init_weight(self.cell_id, LoRAType::WeightDecay, device);

        self.param_learning_rate_lora
            .init_weight(self.cell_id, LoRAType::LearningRate, device);

        self.param_output_gate_lora
            .init_weight(self.cell_id, LoRAType::LearningRate, device);

        if let Some(ref mut value_residual_lora) = self.param_value_residual_lora {
            value_residual_lora.init_weight(self.cell_id, LoRAType::ValueResidual, device);
        }

        constant_init(&mut self.param_key_removal, 0.71);
        constant_init(&mut self.param_key_replacement, 1.02);
        constant_init(&mut self.param_receptance_key_bonus, -0.04);

        if let Some(ref mut gamma) = self.group_norm.gamma {
            let layer_scale = ((1 + self.cell_id) as f64 / self.num_cells as f64).powf(0.7);

            constant_init(gamma, layer_scale);
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        v_first: Tensor<B, 3>,
        shift_embedded: Tensor<B, 2>,
        state: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 4>)
    where
        B: Wkv7PretrainBackend,
    {
        let [batch_size_per_device, context_length, embedded_dim] = x.dims();

        let (num_heads, head_size) = (self.num_heads, self.head_size);

        let x_state_out = x
            .clone()
            .slice([
                0..batch_size_per_device,
                (context_length - 1)..context_length,
            ])
            .squeeze_dim(1);

        let (
            time_shifted_diff,
            v_first,
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
        ) = self.weight_prepare(x.clone(), v_first.clone(), shift_embedded.clone());

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

        let x_gate = x.clone() + time_shifted_diff * self.param_gate.val();

        let gate = self.param_output_gate_lora.forward(x_gate);

        let wkv_out = wkv7_pretrain_forward(
            wkv_weight_decay_input.clone(),
            wkv_receptance_input.clone(),
            wkv_key_input.clone(),
            wkv_value_input.clone(),
            wkv_removal_input.clone(),
            wkv_replacement_input.clone(),
            16,
        );

        let out = wkv_out
            .output
            .reshape([batch_size_per_device, context_length, embedded_dim]);

        let _current_vk_state = state.clone();

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

        let bonus: Tensor<B, 3> =
            bonus.reshape([batch_size_per_device, context_length, embedded_dim]);

        let out_gated = (out_normalized + bonus) * gate;

        let out = self.projection_output.forward(out_gated);

        (out, v_first, x_state_out, state)
    }

    pub fn weight_prepare(
        &self,
        x: Tensor<B, 3>,
        v_first: Tensor<B, 3>,
        x_state: Tensor<B, 2>,
    ) -> TimeMixerWeightPrepareOutput<B> {
        // Paper equations implemented:
        // 355: x^{square}_t = lerp(x_t, x_{t-1}, mu_{square})  -- Time shifting
        // 356: a_t = sigmoid(loramlp_a(Identity, x^a_t, bias=True))  -- In-context
        // learning rate 357: k_t = x^k_t W_k  -- Key precursor
        // 358: kappa_t = k_t ⊙ xi  -- Removal key (before normalization)
        // 359: tilde_k_t = k_t ⊙ lerp(1, a_t, alpha)  -- Replacement key
        // 360: nu_t = sigmoid(loramlp_nu(Identity, x^v_t, bias=True))  -- Value
        // residual gate 361-366: v_t computation with residual mixing
        // 367: d_t = loramlp_d(tanh, x^d_t, bias=True)  -- Decay precursor
        // 368: w_t = exp(-e^{-0.5} sigmoid(d_t))  -- Decay
        // 369: r_t = x^r_t W_r  -- Receptance
        // 370: g_t = loramlp_g(sigmoid, x^g_t, bias=False)  -- RWKV gate
        let [batch_size, sequence_length, channel_dim] = x.dims();

        let (num_heads, head_size) = (self.num_heads, self.head_size);

        let time_shifted_diff = token_shift(x.clone(), x_state) - x.clone();

        let receptance_input = x.clone() + time_shifted_diff.clone() * self.param_receptance.val();

        let weight_decay_input =
            x.clone() + time_shifted_diff.clone() * self.param_weight_decay.val();

        let key_input = x.clone() + time_shifted_diff.clone() * self.param_key.val();

        let value_input = x.clone() + time_shifted_diff.clone() * self.param_value.val();

        let learning_rate_input =
            x.clone() + time_shifted_diff.clone() * self.param_learning_rate.val();

        let receptance = self.projection_receptance.forward(receptance_input);

        let key_precursor = self.projection_key.forward(key_input);

        let value_precursor = self.projection_value.forward(value_input.clone());

        let v_first = if self.cell_id == 0 {
            value_precursor.clone()
        } else {
            v_first
        };

        let learning_rate = sigmoid(self.param_learning_rate_lora.forward(learning_rate_input));

        let ones_like_k = Tensor::ones_like(&key_precursor);

        let alpha_modulated = ones_like_k.clone()
            + (learning_rate.clone() - ones_like_k) * self.param_key_replacement.val();

        let replacement_key = key_precursor.clone() * alpha_modulated;

        let value = if self.cell_id != 0 {
            let nu_t = sigmoid(
                self.param_value_residual_lora
                    .clone()
                    .unwrap()
                    .forward(value_input),
            );

            lerp(value_precursor, v_first.clone(), nu_t)
        } else {
            value_precursor
        };

        let weight_decay_lora_result = self.param_weight_decay_lora.forward(weight_decay_input);

        let weight_decay = -softplus(-weight_decay_lora_result, 1.0) - 0.5;

        let removal_key = key_precursor * self.param_key_removal.val();

        let removal_key_reshaped =
            removal_key.reshape([batch_size, sequence_length, num_heads, head_size]);

        let removal_key_normalized = normalize(removal_key_reshaped, 2.0, -1, 1e-12).reshape([
            batch_size,
            sequence_length,
            channel_dim,
        ]);

        let replacement = removal_key_normalized.clone() * learning_rate;

        (
            time_shifted_diff,
            v_first,
            receptance,
            weight_decay,
            replacement_key,
            value,
            -removal_key_normalized,
            replacement,
        )
    }
}
