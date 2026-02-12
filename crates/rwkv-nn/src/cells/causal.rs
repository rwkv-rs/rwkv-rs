use crate::kernels::wkv7_common::Wkv7Kernel;
use crate::modules::channel_mixer::ChannelMixerIO;
use crate::modules::time_mixer::TimeMixerIO;
use crate::modules::{
    channel_mixer::{ChannelMixer, ChannelMixerConfig},
    time_mixer::{TimeMixer, TimeMixerConfig},
};
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

    pub fn forward<K: Wkv7Kernel<B>>(
        &self,
        multi_causal_cells_input: MultiCausalCellsIO<B>,
    ) -> MultiCausalCellsIO<B> {
        let [batch_size, _context_length, _embedded_dim] =
            multi_causal_cells_input.embedded_context.dims();

        let mut embedded_context = multi_causal_cells_input.embedded_context;
        let mut value_from_first_cell = Tensor::zeros_like(&embedded_context);

        let mut embedded_token_shift_for_time_mix =
            multi_causal_cells_input.embedded_token_shift_for_time_mix;
        let mut state = multi_causal_cells_input.state;
        let mut embedded_token_shift_for_channel_mix =
            multi_causal_cells_input.embedded_token_shift_for_channel_mix;

        for (cell_id, cell) in self.cells.iter().enumerate() {
            let state_of_the_cell = match state.as_ref() {
                Some(state) => Some(
                    state
                        .clone()
                        .slice([
                            0..batch_size,
                            cell_id..cell_id + 1,
                            0..self.num_heads,
                            0..self.head_size,
                            0..self.head_size,
                        ])
                        .squeeze_dim(1),
                ),
                None => None,
            };

            let cell_time_shift = match embedded_token_shift_for_time_mix.as_ref() {
                Some(x) => Some(
                    x.clone()
                        .slice([0..batch_size, cell_id..cell_id + 1, 0..self.embedded_dim])
                        .squeeze_dim(1),
                ),
                None => None,
            };

            let cell_channel_shift = match embedded_token_shift_for_channel_mix.as_ref() {
                Some(x) => Some(
                    x.clone()
                        .slice([0..batch_size, cell_id..cell_id + 1, 0..self.embedded_dim])
                        .squeeze_dim(1),
                ),
                None => None,
            };

            let cell_output = cell.forward::<K>(CausalCellIO {
                embedded_context,
                value_from_first_cell,
                state: state_of_the_cell,
                embedded_token_shift_for_time_mix: cell_time_shift,
                embedded_token_shift_for_channel_mix: cell_channel_shift,
            });

            let CausalCellIO {
                embedded_context: next_embedded_context,
                value_from_first_cell: next_value_from_first_cell,
                embedded_token_shift_for_time_mix: next_embedded_token_shift_for_time_mix,
                state: next_state,
                embedded_token_shift_for_channel_mix: next_embedded_token_shift_for_channel_mix,
            } = cell_output;

            embedded_context = next_embedded_context;
            value_from_first_cell = next_value_from_first_cell;

            embedded_token_shift_for_time_mix = match embedded_token_shift_for_time_mix {
                Some(embedded_token_shift_for_time_mix) => Some(
                    embedded_token_shift_for_time_mix.slice_assign(
                        [0..batch_size, cell_id..cell_id + 1, 0..self.embedded_dim],
                        next_embedded_token_shift_for_time_mix
                            .unwrap()
                            .unsqueeze_dim(1),
                    ),
                ),
                None => None,
            };

            state = match state {
                Some(state) => Some(state.slice_assign(
                    [
                        0..batch_size,
                        cell_id..cell_id + 1,
                        0..self.num_heads,
                        0..self.head_size,
                        0..self.head_size,
                    ],
                    next_state.unwrap().unsqueeze_dim(1),
                )),
                None => None,
            };

            embedded_token_shift_for_channel_mix = match embedded_token_shift_for_channel_mix {
                Some(embedded_token_shift_for_channel_mix) => {
                    Some(embedded_token_shift_for_channel_mix.slice_assign(
                        [0..batch_size, cell_id..cell_id + 1, 0..self.embedded_dim],
                        next_embedded_token_shift_for_channel_mix.unwrap().unsqueeze_dim(1),
                    ))
                }
                None => None,
            };
        }

        MultiCausalCellsIO {
            embedded_context,
            embedded_token_shift_for_time_mix,
            state,
            embedded_token_shift_for_channel_mix,
        }
    }
}

pub struct MultiCausalCellsIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>, // [batch_size, context_length, embedded_dim]
    pub embedded_token_shift_for_time_mix: Option<Tensor<B, 3>>, // [batch_size, num_cells, embedded_dim]
    pub state: Option<Tensor<B, 5>>,    // [batch_size, num_cells, num_heads, head_size, head_size]
    pub embedded_token_shift_for_channel_mix: Option<Tensor<B, 3>>, // [batch_size, num_cells, embedded_dim]
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

    pub fn forward<K: Wkv7Kernel<B>>(&self, causal_cell_input: CausalCellIO<B>) -> CausalCellIO<B> {
        let embedded_context = causal_cell_input.embedded_context;

        let embedded_context_normalized = self
            .pre_layer_norm_for_time_mix
            .forward(embedded_context.clone());
        let time_mixer_input = TimeMixerIO {
            embedded_context: embedded_context_normalized,
            value_from_first_cell: causal_cell_input.value_from_first_cell.clone(),
            embedded_token_shift: causal_cell_input.embedded_token_shift_for_time_mix,
            state: causal_cell_input.state,
        };

        let time_mixer_output = self.time_mixer.forward::<K>(time_mixer_input);

        let embedded_context = embedded_context + time_mixer_output.embedded_context;

        let embedded_context_normalized = self
            .pre_layer_norm_for_channel_mix
            .forward(embedded_context.clone());

        let channel_mixer_input = ChannelMixerIO {
            embedded_context: embedded_context_normalized,
            embedded_token_shift: causal_cell_input.embedded_token_shift_for_channel_mix,
        };

        let channel_mixer_output = self.channel_mixer.forward(channel_mixer_input);

        let embedded_context = embedded_context + channel_mixer_output.embedded_context;

        CausalCellIO {
            embedded_context,
            value_from_first_cell: time_mixer_output.value_from_first_cell,
            embedded_token_shift_for_time_mix: time_mixer_output.embedded_token_shift,
            state: time_mixer_output.state,
            embedded_token_shift_for_channel_mix: channel_mixer_output.embedded_token_shift,
        }
    }
}

pub struct CausalCellIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>, // [batch_size, context_len, embedded_dim]
    pub value_from_first_cell: Tensor<B, 3>, // [batch_size, context_len, embedded_dim]
    pub embedded_token_shift_for_time_mix: Option<Tensor<B, 2>>, // [batch_size, embedded_dim]
    pub state: Option<Tensor<B, 4>>,    // [batch_size, num_heads, head_size, head_size]
    pub embedded_token_shift_for_channel_mix: Option<Tensor<B, 2>>, // [batch_size, embedded_dim]
}
