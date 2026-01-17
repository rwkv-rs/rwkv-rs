use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::Tensor,
};
use rwkv_data::mmap::dtype::TokenUnit;

use crate::{
    cells::{
        bidirectional::{BidirectionalCell, BidirectionalCellConfig, BidirectionalCellState},
        causal::CausalCellState,
    },
    functions::init_weights::{orthogonal_init, uniform_init},
    kernels::wkv7::Wkv7Backend,
    layers::embedding::{EmbModule, EmbModuleConfig, TokensOptions},
};

#[derive(Config, Debug)]

pub struct SequenceEmbeddingModelConfig {
    num_cells: usize,
    vocabulary_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl SequenceEmbeddingModelConfig {
    pub fn init<B: Backend, T: TokenUnit>(&self, device: &B::Device) -> SequenceEmbeddingModel<B> {
        SequenceEmbeddingModel {
            embed: EmbModuleConfig::new(self.vocabulary_size, self.embedded_dim)
                .init::<B, T>(device),
            layer_norm_for_first_cell: LayerNormConfig::new(self.embedded_dim).init(device),
            cells: (0..self.num_cells)
                .map(|i| {
                    BidirectionalCellConfig::new(
                        self.num_cells,
                        self.embedded_dim,
                        self.num_heads,
                        self.head_size,
                    )
                    .init(i, device)
                })
                .collect(),
            layer_norm_for_unembed: LayerNormConfig::new(self.embedded_dim).init(device),
            unembed: LinearConfig::new(self.embedded_dim, self.vocabulary_size).init(device),

            num_cells: self.num_cells,
            vocabulary_size: self.vocabulary_size,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]

pub struct SequenceEmbeddingModel<B: Backend> {
    pub embed: EmbModule<B>,
    pub layer_norm_for_first_cell: LayerNorm<B>,
    pub cells: Vec<BidirectionalCell<B>>,
    pub layer_norm_for_unembed: LayerNorm<B>,
    pub unembed: Linear<B>,

    num_cells: usize,
    vocabulary_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> SequenceEmbeddingModel<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        match &mut self.embed {
            EmbModule::Discrete(emb) => {
                let lr_init = 0.45 / self.embedded_dim as f64;

                uniform_init(&mut emb.weight, -lr_init, lr_init);
            }
            EmbModule::Continuous(linear) => {
                let lr_init = 0.45 / self.embedded_dim as f64;

                uniform_init(&mut linear.weight, -lr_init, lr_init);
            }
        }

        orthogonal_init(&mut self.unembed.weight, Some(0.5));

        self.cells
            .iter_mut()
            .for_each(|cell| cell.init_weights(device));
    }

    pub fn forward(
        &self,
        x: TokensOptions<B>,
        s: Vec<BidirectionalCellState<B>>,
        device: &Device<B>,
    ) -> (Tensor<B, 3>, Vec<BidirectionalCellState<B>>)
    where
        B: Wkv7Backend,
    {
        let x = self.embed.forward(x);

        let mut states = if s.is_empty() {
            let batch_size = x.dims()[0];

            (0..self.num_cells)
                .map(|_| {
                    let empty_state = CausalCellState {
                        time_shift_embedded: Tensor::<B, 2>::zeros(
                            [batch_size, self.embedded_dim],
                            device,
                        ),
                        time_mix_state: Tensor::<B, 4, Float>::zeros(
                            [batch_size, self.num_heads, self.head_size, self.head_size],
                            device,
                        ),
                        channel_shift_embedded: Tensor::<B, 2>::zeros(
                            [batch_size, self.embedded_dim],
                            device,
                        ),
                    };

                    BidirectionalCellState {
                        causal_past2future: empty_state.clone(),
                        causal_future2past: empty_state,
                    }
                })
                .collect::<Vec<_>>()
        } else {
            s
        };

        let mut v_first = Tensor::zeros_like(&x);

        let mut x = self.layer_norm_for_first_cell.forward(x);

        for (cell_id, cell) in self.cells.iter().enumerate() {
            let (new_x, new_v_first, new_state) =
                cell.forward(x, v_first, states[cell_id].clone(), device);

            x = new_x;

            v_first = new_v_first;

            states[cell_id] = new_state;
        }

        let x = self.layer_norm_for_unembed.forward(x);

        let x = self.unembed.forward(x);

        (x, states)
    }
}
