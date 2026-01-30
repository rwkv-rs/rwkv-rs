use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::{Backend, Tensor},
    tensor::activation::sigmoid,
};

use crate::{
    cells::causal::{CausalCell, CausalCellConfig, CausalCellState},
    functions::lerp::lerp,
    kernels::wkv7::Wkv7Backend,
};

#[derive(Config, Debug)]
pub struct BidirectionalCellConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl BidirectionalCellConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> BidirectionalCell<B> {
        BidirectionalCell {
            causal_past2future: CausalCellConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(cell_id, device),
            causal_future2past: CausalCellConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(cell_id, device),
            gate_x: LinearConfig::new(self.embedded_dim * 2, self.embedded_dim).init(device),
            gate_v_first: LinearConfig::new(self.embedded_dim * 2, self.embedded_dim).init(device),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct BidirectionalCell<B: Backend> {
    pub causal_past2future: CausalCell<B>,
    pub causal_future2past: CausalCell<B>,
    pub gate_x: Linear<B>,
    pub gate_v_first: Linear<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
    cell_id: usize,
}

impl<B: Backend> BidirectionalCell<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.causal_past2future.init_weights(device);

        self.causal_future2past.init_weights(device);
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        v_first: Tensor<B, 3>,
        state: BidirectionalCellState<B>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, BidirectionalCellState<B>)
    where
        B: Wkv7Backend,
    {
        let (x_past2future, new_v_first_past2future, new_state_past2future) = self
            .causal_past2future
            .forward(x.clone(), v_first.clone(), state.causal_past2future);

        let (x_future2past, new_v_first_future2past, new_state_future2past) =
            self.causal_future2past.forward(
                x.flip([1]),
                v_first.clone(),
                state.causal_future2past,
            );

        let x_future2past_rev = x_future2past.flip([1]);

        let gate_input_x = Tensor::cat(vec![x_past2future.clone(), x_future2past_rev.clone()], 2);

        let ratio_x = sigmoid(self.gate_x.forward(gate_input_x));

        let x = lerp(x_past2future, x_future2past_rev, ratio_x);

        let v_first = if self.cell_id == 0 {
            let new_v_first_future2past_rev = new_v_first_future2past.flip([1]);

            let gate_input_v_first = Tensor::cat(
                vec![
                    new_v_first_past2future.clone(),
                    new_v_first_future2past_rev.clone(),
                ],
                2,
            );

            let ratio_v_first = sigmoid(self.gate_v_first.forward(gate_input_v_first));

            lerp(
                new_v_first_past2future,
                new_v_first_future2past_rev,
                ratio_v_first,
            )
        } else {
            v_first
        };

        let bidirectional_cell_state = BidirectionalCellState {
            causal_past2future: new_state_past2future,
            causal_future2past: new_state_future2past,
        };

        (x, v_first, bidirectional_cell_state)
    }
}

#[derive(Clone, Debug)]

pub struct BidirectionalCellState<B: Backend> {
    pub causal_past2future: CausalCellState<B>,
    pub causal_future2past: CausalCellState<B>,
}
