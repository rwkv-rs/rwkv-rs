// This is a basic text classification model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

use rwkv::custom::config::Config;
use rwkv::custom::module::Module;
use rwkv::custom::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use rwkv::custom::nn::loss::CrossEntropyLossConfig;
use rwkv::custom::Tensor;
use rwkv::custom::tensor::backend::{AutodiffBackend, Backend};
use rwkv::custom::train::{InferenceStep, TrainOutput, TrainStep};
use rwkv::nn::cells::causal::{MultiCausalCells, MultiCausalCellsConfig, MultiCausalCellsIO};
use rwkv::nn::functions::init_weights::{orthogonal_init, uniform_init};
use rwkv::nn::kernels::l2wrap::{l2wrap, L2WrapBackend};
use rwkv::nn::kernels::wkv7_common::{KernelPretrain, Wkv7Backend, Wkv7Kernel};
use rwkv::nn::modules::time_mixer::param_state::{StateModule, StateModuleConfig};
use rwkv::train::learner::next_token_prediction::NextTokenPredictionOutput;
use crate::data::batcher::AutoRegressiveBatch;

rwkv::custom_mode!();


#[derive(Config, Debug)]
pub struct AutoRegressiveModelConfig {
    num_cells: usize,
    vocabulary_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl AutoRegressiveModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoRegressiveModel<B> {
        AutoRegressiveModel {
            embed: EmbeddingConfig::new(self.vocabulary_size, self.embedded_dim).init(device),
            layer_norm_for_first_cell: LayerNormConfig::new(self.embedded_dim).init(device),
            cells: MultiCausalCellsConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(device),
            layer_norm_for_unembed: LayerNormConfig::new(self.embedded_dim).init(device),
            unembed: LinearConfig::new(self.embedded_dim, self.vocabulary_size)
                .with_bias(false)
                .init(device),
            state: StateModuleConfig::new(self.num_cells, self.num_heads, self.head_size)
                .init(device),

            num_cells: self.num_cells,
            vocabulary_size: self.vocabulary_size,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct AutoRegressiveModel<B: Backend> {
    pub embed: Embedding<B>,
    pub layer_norm_for_first_cell: LayerNorm<B>,
    pub cells: MultiCausalCells<B>,
    pub layer_norm_for_unembed: LayerNorm<B>,
    pub unembed: Linear<B>,
    pub state: StateModule<B>,

    num_cells: usize,
    vocabulary_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> AutoRegressiveModel<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        uniform_init(&mut self.embed.weight, -1e-4, 1e-4);

        if self.vocabulary_size > self.embedded_dim {
            orthogonal_init(
                &mut self.unembed.weight,
                Some(0.5 * (self.vocabulary_size as f32 / self.embedded_dim as f32).sqrt()),
            );
        } else {
            orthogonal_init(&mut self.unembed.weight, Some(0.5));
        }

        self.cells.init_weights(device);
    }

    pub fn forward<K: Wkv7Kernel<B>>(
        &self,
        item: AutoRegressiveBatch<B>,
        embedded_token_shift_for_time_mix: Option<Tensor<B, 3>>,
        state: Option<Tensor<B, 5>>,
        embedded_token_shift_for_channel_mix: Option<Tensor<B, 3>>,
    ) -> NextTokenPredictionOutput<B>
    where
        B: L2WrapBackend,
    {
        let [batch_size, context_length] = item.inputs.dims();
        let device = &self.embed.devices()[0];

        let inputs = item.inputs.to_device(device);
        let targets = item.targets.to_device(device);

        let embedded_context = self.embed.forward(inputs);
        let embedded_context_normalized = self.layer_norm_for_first_cell.forward(embedded_context);

        let multi_causal_cells_input = MultiCausalCellsIO {
            embedded_context: embedded_context_normalized,
            embedded_token_shift_for_time_mix,
            state,
            embedded_token_shift_for_channel_mix,
        };
        let multi_causal_cells_output = self.cells.forward::<K>(multi_causal_cells_input);
        let embedded_context = multi_causal_cells_output.embedded_context;
        let embedded_context_normalized = self.layer_norm_for_unembed.forward(embedded_context);
        let logits = self.unembed.forward(embedded_context_normalized);

        let num_tokens_per_batch = batch_size * context_length;
        let logits_flat = logits.reshape([
            num_tokens_per_batch, self.vocabulary_size
        ]);
        let targets_flat = targets.reshape([
            num_tokens_per_batch,
        ]);
        let loss = l2wrap(
            CrossEntropyLossConfig::new()
                .init(&logits_flat.device())
                .forward(logits_flat.clone(), targets_flat),
            logits_flat,
            num_tokens_per_batch,
        );

        // Return the output and loss
        NextTokenPredictionOutput { loss }
    }
}



impl<B: AutodiffBackend + Wkv7Backend + L2WrapBackend> TrainStep for AutoRegressiveModel<B> {
    type Input = AutoRegressiveBatch<B>;
    type Output = NextTokenPredictionOutput<B>;

    fn step(
        &self,
        item: AutoRegressiveBatch<B>,
    ) -> TrainOutput<NextTokenPredictionOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        let item = self.forward::<KernelPretrain>(item, None, None, None);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}


/// Define validation step
impl<B: Backend + Wkv7Backend + L2WrapBackend> InferenceStep for AutoRegressiveModel<B> {
    type Input = AutoRegressiveBatch<B>;
    type Output = NextTokenPredictionOutput<B>;

    fn step(&self, item: AutoRegressiveBatch<B>) -> NextTokenPredictionOutput<B> {
        // Run forward pass and return the output
        self.forward::<KernelPretrain>(item, None, None, None)
    }
}