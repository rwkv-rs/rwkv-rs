use crate::{
    kernels::wkv7_pretrain::Wkv7PretrainBackend,
    modules::{
        channel_mixer::{ChannelMixer, ChannelMixerConfig},
        time_mixer::{TimeMixer, TimeMixerConfig},
    },
};
use burn::prelude::Float;
use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    prelude::{Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct MultiCausalCellsConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl MultiCausalCellsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiCausalCells<B> {
        MultiCausalCells {
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

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiCausalCells<B: Backend> {
    pub cells: Vec<CausalCell<B>>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> MultiCausalCells<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.cells
            .iter_mut()
            .for_each(|cell| cell.init_weights(device));
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        s: Option<Vec<CausalCellState<B>>>,
    ) -> (Tensor<B, 3>, Vec<CausalCellState<B>>)
    where
        B: Wkv7PretrainBackend,
    {
        let mut states = s.unwrap_or_else(|| self.init_states(x.dims()[0], &x.device()));

        let mut x = x;
        let mut v_first = Tensor::zeros_like(&x);

        for (cell_id, cell) in self.cells.iter().enumerate() {
            let (new_x, new_v_first, new_state) = cell.forward(x, v_first, states[cell_id].clone());

            x = new_x;

            v_first = new_v_first;

            states[cell_id] = new_state;
        }
        (x, states)
    }

    fn init_states(&self, batch_size: usize, device: &B::Device) -> Vec<CausalCellState<B>> {
        (0..self.num_cells)
            .map(|_| CausalCellState {
                time_shift_embedded: Tensor::<B, 2>::zeros([batch_size, self.embedded_dim], device),
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
    }
}

#[derive(Config, Debug)]
pub struct CausalCellConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl CausalCellConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> CausalCell<B> {
        CausalCell {
            pre_layer_norm_for_time_mix: LayerNormConfig::new(self.embedded_dim).init(device),
            time_mixer: TimeMixerConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(cell_id, device),
            pre_layer_norm_for_channel_mix: LayerNormConfig::new(self.embedded_dim).init(device),
            channel_mixer: ChannelMixerConfig::new(self.num_cells, self.embedded_dim)
                .init(cell_id, device),
            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalCell<B: Backend> {
    pub pre_layer_norm_for_time_mix: LayerNorm<B>,
    pub pre_layer_norm_for_channel_mix: LayerNorm<B>,
    pub time_mixer: TimeMixer<B>,
    pub channel_mixer: ChannelMixer<B>,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> CausalCell<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.time_mixer.init_weights(device);

        self.channel_mixer.init_weights(device);
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        v_first: Tensor<B, 3>,
        state: CausalCellState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, CausalCellState<B>)
    where
        B: Wkv7PretrainBackend,
    {
        let x = x;

        let x_time_mix_input = self.pre_layer_norm_for_time_mix.forward(x.clone());

        let (x_time_mix_output, v_first, time_shift_embedded, time_mix_state) =
            self.time_mixer.forward(
                x_time_mix_input,
                v_first,
                state.time_shift_embedded,
                state.time_mix_state,
            );

        let x = x + x_time_mix_output;

        let x_channel_mix_input = self.pre_layer_norm_for_channel_mix.forward(x.clone());

        let (x_channel_mix_output, channel_shift_embedded) = self
            .channel_mixer
            .forward(x_channel_mix_input, state.channel_shift_embedded);

        let x = x + x_channel_mix_output;

        let causal_cell_state = CausalCellState {
            time_shift_embedded,
            time_mix_state,
            channel_shift_embedded,
        };

        (x, v_first, causal_cell_state)
    }
}

#[derive(Clone, Debug)]
pub struct CausalCellState<B: Backend> {
    pub time_shift_embedded: Tensor<B, 2>,
    pub time_mix_state: Tensor<B, 4>,
    pub channel_shift_embedded: Tensor<B, 2>,
}
