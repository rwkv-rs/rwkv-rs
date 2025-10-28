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

#[cfg(test)]

mod tests {

    use super::*;
    use crate::utils::test_tools::*;

    #[test]

    fn test_preln() {
        let device = &get_test_device::<TestBackend>();
        let model = get_test_model(device);

        let mut preln_outputs = vec![];

        let mut expected_preln_outputs = vec![];

        let mut module_names = vec![];

        for &cell_id in &[0, 1, 11] {
            let mut input = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_input", cell_id).as_str(),
                device,
            );

            let cell = model.cells[cell_id].clone();
            if cell_id == 0 {
                input = model.layer_norm_for_first_cell.forward(input);
            }

            let preln_output = cell.pre_layer_norm_for_time_mix.forward(input);

            let expected_preln_output = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_att_input", cell_id).as_str(),
                device,
            );

            preln_outputs.push(preln_output);
            expected_preln_outputs.push(expected_preln_output);
            module_names.push(format!("cell_{}_preln", cell_id));
        }

        assert_closeness_multi(
            preln_outputs,
            expected_preln_outputs,
            module_names,
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
    }

    #[test]
    fn test_channel_mix() {
        let device = &get_test_device::<TestBackend>();
        let model = get_test_model(device);

        let mut channel_mix_outputs = vec![];
        let mut expected_channel_mix_outputs = vec![];
        let mut module_names = vec![];

        for &cell_id in &[0, 1, 11] {
            let input = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_ffn_input_x", cell_id).as_str(),
                device,
            );
            let cell = model.cells[cell_id].clone();

            let (channel_mix_output_x, _) = cell.channel_mixer.forward(
                input.clone(),
                Tensor::<TestBackend, 2>::zeros([1, TEST_EMBEDDED_DIM], &device),
            );

            let expected_channel_mix_output_x = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_ffn_output_x", cell_id).as_str(),
                device,
            );

            channel_mix_outputs.push(channel_mix_output_x);
            expected_channel_mix_outputs.push(expected_channel_mix_output_x);
            module_names.push(format!("cell_{}_channel_mix", cell_id));
        }

        assert_closeness_multi(
            channel_mix_outputs,
            expected_channel_mix_outputs,
            module_names,
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
    }
}
