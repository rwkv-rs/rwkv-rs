use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

#[derive(Config, Debug)]

pub struct StateAdapterModelConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
    hidden_size: usize,
}

impl StateAdapterModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StateAdapterModel<B> {
        let mut proj = vec![];

        let mut norm = vec![];

        for _head_index in 0..self.num_heads {
            proj.push(LinearConfig::new(self.head_size.pow(2), self.hidden_size).init(device));

            norm.push(LayerNormConfig::new(self.hidden_size).init(device))
        }

        StateAdapterModel {
            proj_head: proj,
            norm_head: norm,

            proj_state: LinearConfig::new(
                self.num_cells * self.num_heads * self.hidden_size,
                self.embedded_dim,
            )
            .init(device),

            norm_state: LayerNormConfig::new(self.embedded_dim).init(device),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
            hidden_size: self.hidden_size,
        }
    }
}

#[derive(Module, Debug)]

pub struct StateAdapterModel<B: Backend> {
    pub proj_head: Vec<Linear<B>>,
    pub norm_head: Vec<LayerNorm<B>>,
    pub proj_state: Linear<B>,
    pub norm_state: LayerNorm<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
    hidden_size: usize,
}

impl<B: Backend> StateAdapterModel<B> {
    pub fn forward(&self, time_mix_state: Tensor<B, 4>) -> Tensor<B, 2> {
        let state =
            time_mix_state.reshape([self.num_cells * self.num_heads, self.head_size.pow(2)]);

        let mut embedded_of_heads = Vec::with_capacity(self.num_cells * self.num_heads);

        for head_index in 0..self.num_cells * self.num_heads {
            let state_of_head = state.clone().slice(s![head_index, ..]);

            let projected = self.proj_head[head_index].forward(state_of_head);

            let normalized = self.norm_head[head_index].forward(projected);

            embedded_of_heads.push(normalized);
        }

        let embedded = Tensor::cat(embedded_of_heads, 0);

        let projected = self.proj_state.forward(embedded);

        let normalized = self.norm_state.forward(projected);

        normalized
    }
}
