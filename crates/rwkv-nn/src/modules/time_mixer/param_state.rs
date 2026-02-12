use burn::{config::Config, module::{Module, Param}, prelude::*};

#[derive(Config, Debug)]
pub struct StateModuleConfig {
    num_cells: usize,
    num_heads: usize,
    head_size: usize,
}

impl StateModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StateModule<B> {
        let initial_state = Tensor::zeros(
            [self.num_cells, self.num_heads, self.head_size, self.head_size],
            device,
        );

        StateModule {
            state: Param::from_tensor(initial_state),
            num_cells: self.num_cells,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct StateModule<B: Backend> {
    pub state: Param<Tensor<B, 4>>,

    num_cells: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> StateModule<B> {
    pub fn get_state(&self, batch_size: usize) -> Tensor<B, 5> {
        let state_batch_size_one: Tensor<B, 5> = self.state.val().unsqueeze_dim(0);
        let mut state = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            state.push(state_batch_size_one.clone());
        }

        Tensor::cat(state, 0)
    }
}
