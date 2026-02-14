use burn::{
    config::Config,
    module::{Module, Param},
    prelude::*,
};

#[derive(Config, Debug)]
pub struct StateModuleConfig {
    num_cells: usize,
    num_heads: usize,
    head_size: usize,
}

impl StateModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StateModule<B> {
        StateModule {
            state: (0..self.num_cells)
                .map(|_| {
                    Param::from_tensor(Tensor::zeros(
                        [self.num_heads, self.head_size, self.head_size],
                        device,
                    ))
                })
                .collect(),
            num_cells: self.num_cells,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct StateModule<B: Backend> {
    pub state: Vec<Param<Tensor<B, 3>>>,

    num_cells: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> StateModule<B> {
    pub fn get_state(&self, batch_size: usize) -> Vec<Tensor<B, 4>> {
        debug_assert_eq!(self.state.len(), self.num_cells);
        let mut state = Vec::with_capacity(self.state.len());

        for param_state_for_one_cell in &self.state {
            let state_for_one_cell_one_batch: Tensor<B, 4> =
                param_state_for_one_cell.val().unsqueeze_dim(0);
            let mut state_for_one_cell = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                state_for_one_cell.push(state_for_one_cell_one_batch.clone());
            }
            state.push(Tensor::cat(state_for_one_cell, 0));
        }

        state
    }
}
