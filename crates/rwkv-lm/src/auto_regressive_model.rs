use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::Tensor,
};
use rwkv_data::mmap::dtype::TokenUnit;

use crate::{
    cells::causal::{CausalCell, CausalCellConfig, CausalCellState},
    functions::init_weights::{orthogonal_init, uniform_init},
    kernels::wkv7::Wkv7Backend,
    layers::embedding::{EmbModule, EmbModuleConfig, TokensOptions},
};

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
            embed: EmbModuleConfig::new(self.vocabulary_size, self.embedded_dim)
                .init::<B, T>(device),
            layer_norm_for_first_cell: LayerNormConfig::new(self.embedded_dim).init(device),
            cells: (0..self.num_cells)
                .map(|i| {
                    CausalCellConfig::new(
                        self.num_cells,
                        self.embedded_dim,
                        self.num_heads,
                        self.head_size,
                    )
                    .init(i, device)
                })
                .collect(),
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
    pub embed: EmbModule<B>,
    pub layer_norm_for_first_cell: LayerNorm<B>,
    pub cells: Vec<CausalCell<B>>,
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
        match &mut self.embed {
            EmbModule::Discrete(emb) => {
                uniform_init(&mut emb.weight, -1e-4, 1e-4);
            }
            EmbModule::Continuous(_linear) => {}
        }

        if self.vocabulary_size > self.embedded_dim {
            orthogonal_init(
                &mut self.unembed.weight,
                Some(0.5 * (self.vocabulary_size as f32 / self.embedded_dim as f32).sqrt()),
            );
        } else {
            orthogonal_init(&mut self.unembed.weight, Some(0.5));
        }

        self.cells
            .iter_mut()
            .for_each(|cell| cell.init_weights(device));
    }

    pub fn forward(
        &self,
        x: TokensOptions<B>,
        s: Vec<CausalCellState<B>>,
    ) -> (Tensor<B, 3>, Vec<CausalCellState<B>>)
    where
        B: Wkv7Backend,
    {
        let device = &x.device();

        let x = self.embed.forward(x);

        let mut states = if s.is_empty() {
            let batch_size = x.dims()[0];

            (0..self.num_cells)
                .map(|_| CausalCellState {
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
