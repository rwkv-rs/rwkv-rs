use burn::{
    Tensor,
    prelude::Backend,
    tensor::{Int, TensorData},
};
use itertools::Itertools;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,

    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub penalty_decay: f32,

    pub max_new_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: -1,
            top_p: 0.3,

            presence_penalty: 0.5,
            repetition_penalty: 0.5,
            penalty_decay: 0.996,

            max_new_tokens: 256,
        }
    }
}

impl SamplingConfig {
    pub fn check(&mut self, vocab_size: i32) {
        self.temperature = self.temperature.clamp(0.001, 1000.0);

        if self.top_k <= 0 || self.top_k > vocab_size {
            self.top_k = vocab_size;
        }

        if !(0f32..=1f32).contains(&self.top_p) {
            self.top_p = 1f32;
        }

        if self.top_p == 0f32 {
            // Match rapid-sampling behavior: deterministic argmax when top_p==0.
            self.top_k = 1;
            self.top_p = 1f32;
        }
    }
    pub fn penalties_enabled(&self) -> bool {
        self.presence_penalty != 0.0 || self.repetition_penalty != 0.0
    }
}

pub fn sampling_configs_to_tensor<B: Backend>(
    sampling_configs: &[SamplingConfig],
    vocab_size: usize,
    device: &B::Device,
) -> SamplingConfigsTensor<B> {
    let (
        inv_temperatures,
        top_ks,
        top_ps,
        presence_penalties,
        repetition_penalties,
        penalties_decay,
    ): (Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) = sampling_configs
        .iter()
        .copied()
        .map(|mut sampling_config| {
            sampling_config.check(vocab_size as i32);
            (
                1.0 / sampling_config.temperature,
                sampling_config.top_k,
                sampling_config.top_p,
                sampling_config.presence_penalty,
                sampling_config.repetition_penalty,
                sampling_config.penalty_decay,
            )
        })
        .multiunzip();

    let float_tensor_from_vec = |float_values: Vec<f32>| {
        let batch_size = float_values.len();
        Tensor::<B, 1>::from_data(TensorData::new(float_values, [batch_size]), device)
    };

    let int_tensor_from_vec = |int_values: Vec<i32>| {
        let batch_size = int_values.len();
        Tensor::<B, 1, Int>::from_data(TensorData::new(int_values, [batch_size]), device)
    };

    SamplingConfigsTensor {
        inv_temperatures: float_tensor_from_vec(inv_temperatures),
        top_ks: int_tensor_from_vec(top_ks),
        top_ps: float_tensor_from_vec(top_ps),
        presence_penalties: float_tensor_from_vec(presence_penalties),
        repetition_penalties: float_tensor_from_vec(repetition_penalties),
        penalties_decay: float_tensor_from_vec(penalties_decay),
    }
}

pub struct SamplingConfigsTensor<B: Backend> {
    pub inv_temperatures: Tensor<B, 1>,
    pub top_ks: Tensor<B, 1, Int>,
    pub top_ps: Tensor<B, 1>,
    pub presence_penalties: Tensor<B, 1>,
    pub repetition_penalties: Tensor<B, 1>,
    pub penalties_decay: Tensor<B, 1>,
}
