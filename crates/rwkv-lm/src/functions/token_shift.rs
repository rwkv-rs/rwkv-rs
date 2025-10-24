use burn::{Tensor, prelude::Backend};

pub fn token_shift<B: Backend>(x: Tensor<B, 3>, shift_embedded: Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, t, _] = x.dims();

    Tensor::cat(
        vec![
            shift_embedded.unsqueeze_dim(1),
            x.clone().slice([0..b, 0..(t - 1)]),
        ],
        1,
    )
}

#[cfg(test)]

mod tests {

    use super::*;
    use crate::utils::test_tools::*;

    #[test]

    fn test_time_shift() {
        let model = get_global_model();

        let device = get_global_device();

        let time_shift_input_x = load_expected_f32::<3>("block_0_att_time_shift_input_x");

        let [batch_size, _sequence_length, channel_dim] = time_shift_input_x.dims();

        let time_shift_input_state: Tensor<TestAutodiffBackend, 2> =
            Tensor::zeros([batch_size, channel_dim], &device);

        let shifted_x = token_shift(time_shift_input_x, time_shift_input_state);

        let expected_time_shift_output = load_expected_f32::<3>("block_0_att_time_shift_output_x");

        assert_closeness(
            &shifted_x,
            &expected_time_shift_output,
            "block_0_time_mix_time_shift_output_x",
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
    }
}
