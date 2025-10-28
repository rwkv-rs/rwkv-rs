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
        let device = &get_test_device::<TestBackend>();

        let time_shift_input_x =
            load_expected_f32::<TestBackend, 3>("block_0_att_time_shift_input_x", device);

        let time_shift_input_state: Tensor<TestBackend, 2> =
            Tensor::zeros([TEST_BATCH_SIZE, TEST_CONTEXT_LENGTH], device);

        let shifted_x = token_shift(time_shift_input_x, time_shift_input_state);

        let expected_time_shift_output =
            load_expected_f32::<TestBackend, 3>("block_0_att_time_shift_output_x", device);

        assert_closeness(
            &shifted_x,
            &expected_time_shift_output,
            "block_0_time_mix_time_shift_output_x",
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
    }
}
