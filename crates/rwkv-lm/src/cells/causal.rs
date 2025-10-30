use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    prelude::{Backend, Tensor},
};

use crate::{
    kernels::wkv7::Wkv7Backend,
    modules::{
        channel_mixer::{ChannelMixer, ChannelMixerConfig},
        time_mixer::{TimeMixer, TimeMixerConfig},
    },
};

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
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, CausalCellState<B>)
    where
        B: Wkv7Backend,
    {
        let x = x;

        let x_time_mix_input = self.pre_layer_norm_for_time_mix.forward(x.clone());

        let (x_time_mix_output, v_first, time_shift_embedded, time_mix_state) =
            self.time_mixer.forward(
                x_time_mix_input,
                v_first,
                state.time_shift_embedded,
                state.time_mix_state,
                device,
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
