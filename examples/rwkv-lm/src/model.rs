// This is a basic text classification model implemented in Rust using the Burn framework.
// It uses a Transformer as the base model and applies Linear and Embedding layers.
// The model is then trained using Cross-Entropy loss. It contains methods for model initialization
// (both with and without pre-trained weights), forward pass, inference, training, and validation.

use rwkv::custom::config::Config;
use rwkv::custom::module::Module;
use rwkv::custom::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use rwkv::custom::tensor::backend::Backend;
use rwkv::custom::Tensor;
use rwkv::data::mmap::dtype::TokenUnit;
use rwkv::nn::cells::causal::{CausalCellState, MultiCausalCells, MultiCausalCellsConfig};
use rwkv::nn::functions::init_weights::{orthogonal_init, uniform_init};
use rwkv::nn::kernels::wkv7::Wkv7Backend;
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
    pub fn init<B: Backend, T: TokenUnit>(&self, device: &B::Device) -> AutoRegressiveModel<B> {
        AutoRegressiveModel {
            embed: EmbeddingConfig::new(self.vocabulary_size, self.embedded_dim).init(device),
            layer_norm_for_first_cell: LayerNormConfig::new(self.embedded_dim).init(device),
            cells: MultiCausalCellsConfig::new(
                self.num_cells, self.embedded_dim, self.num_heads, self.head_size
            ).init(device),
            layer_norm_for_unembed: LayerNormConfig::new(self.embedded_dim).init(device),
            unembed: LinearConfig::new(self.embedded_dim, self.vocabulary_size)
                .with_bias(false)
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

    pub fn forward(
        &self,
        item: AutoRegressiveBatch<B>,
    ) -> (Tensor<B, 3>, Vec<CausalCellState<B>>)
    where
        B: Wkv7Backend,
    {
        let x = self.embed.forward(item.inputs);
        let x = self.layer_norm_for_first_cell.forward(x);
        let (x, states) = self.cells.forward(x, None);
        let x = self.layer_norm_for_unembed.forward(x);
        let logits = self.unembed.forward(x);
        (logits, states)
    }
}

//
// impl<B: AutodiffBackend> TrainStep for MyRwkvLM<B> {
//     type Input = AutoRegressiveBatch<B>,
//     type Output = ClassificationOutput<B>,
//     fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
//         let item = self.forward
//     }
// }

