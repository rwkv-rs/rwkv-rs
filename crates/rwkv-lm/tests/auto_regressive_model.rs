mod test_tools;

use burn::{Tensor, prelude::Float, tensor::TensorPrimitive};
use rwkv_lm::{
    functions::token_shift::token_shift, kernels::wkv7::wkv7_forward,
    layers::embedding::TokensOptions,
};
use rwkv_lm::kernels::wkv7::Wkv7Backend;
use test_tools::*;

#[test]
fn test_rwkv7_0p1b_forward() {
    let device = &get_test_device::<TestBackend>();
    let model = get_test_model::<TestBackend>(device);
    let mut all_checks_passed = true;

    let input =
        TokensOptions::SingleUnitIntTokens(load_expected_i64::<TestBackend, 2>("input", device));
    let (output, _) = model.forward(input.clone(), vec![]);
    let expected_output = load_expected_f32::<TestBackend, 3>("output", device);

    let output_pass = check_closeness(
        &output,
        &expected_output,
        "auto_regressive_model",
        MIN_PASS_RATE,
        RELATIVE_ERROR,
    );

    if !output_pass {
        all_checks_passed = false;
        let embed_output = model.embed.forward(input);
        let expected_embed_output = load_expected_f32::<TestBackend, 3>("emb_output", device);

        let embed_pass = check_closeness(
            &embed_output,
            &expected_embed_output,
            "model.embed",
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
        if !embed_pass {
            all_checks_passed = false;
        }
        assert!(embed_pass, "Precision maybe mismatch!");

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

        let preln_pass = check_closeness_multi(
            preln_outputs,
            expected_preln_outputs,
            module_names,
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
        if !preln_pass {
            all_checks_passed = false;
        }

        let expected_v_first =
            load_expected_f32::<TestBackend, 3>("block_0_att_output_v_first", device);

        let mut time_mix_outputs = vec![];
        let mut expected_time_mix_outputs = vec![];
        let mut module_names = vec![];

        for &cell_id in &[0, 1, 11] {
            let input = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_att_input", cell_id).as_str(),
                device,
            );

            let cell = model.cells[cell_id].clone();

            let (time_mix_output_x, _v_first, ..) = cell.time_mixer.forward(
                input.clone(),
                if cell_id == 0 {
                    Tensor::zeros_like(&input)
                } else {
                    expected_v_first.clone()
                },
                Tensor::<TestBackend, 2>::zeros([1, TEST_EMBEDDED_DIM], device),
                Tensor::<TestBackend, 4, Float>::zeros(
                    [1, TEST_NUM_HEADS, TEST_HEAD_SIZE, TEST_HEAD_SIZE],
                    device,
                ),
                device,
            );

            let expected_time_mix_output_x = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_att_output_x", cell_id).as_str(),
                device,
            );

            time_mix_outputs.push(time_mix_output_x);
            expected_time_mix_outputs.push(expected_time_mix_output_x);
            module_names.push(format!("cell_{}_time_mix", cell_id));
        }

        let time_mix_pass = check_closeness_multi(
            time_mix_outputs,
            expected_time_mix_outputs,
            module_names,
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
        if !time_mix_pass {
            all_checks_passed = false;
            let time_shift_input_x =
                load_expected_f32::<TestBackend, 3>("block_0_att_time_shift_input_x", device);

            let time_shift_input_state: Tensor<TestBackend, 2> =
                Tensor::zeros([TEST_BATCH_SIZE, TEST_EMBEDDED_DIM], device);

            let shifted_x = token_shift(time_shift_input_x, time_shift_input_state);

            let expected_time_shift_output =
                load_expected_f32::<TestBackend, 3>("block_0_att_time_shift_output_x", device);

            let time_shift_pass = check_closeness(
                &shifted_x,
                &expected_time_shift_output,
                "block_0_time_mix_time_shift_output_x",
                MIN_PASS_RATE,
                RELATIVE_ERROR,
            );
            if !time_shift_pass {
                all_checks_passed = false;
            }

            let x =
                load_expected_f32::<TestBackend, 3>("block_0_att_weight_prepare_input_x", device);

            let v_first: Tensor<TestBackend, 3> = Tensor::zeros(
                [TEST_BATCH_SIZE, TEST_CONTEXT_LENGTH, TEST_EMBEDDED_DIM],
                device,
            );

            let x_state: Tensor<TestBackend, 2> =
                Tensor::zeros([TEST_BATCH_SIZE, TEST_EMBEDDED_DIM], device);
            let time_mixer = &model.cells[0].time_mixer;

            let (
                _time_shifted_diff,
                v_first,
                receptance,
                weight_decay,
                replacement_key,
                value,
                removal_key_normalized,
                replacement,
            ) = time_mixer.weight_prepare(x, v_first, x_state, device);

            let actual_vec = vec![
                v_first,
                receptance,
                weight_decay,
                replacement_key,
                value,
                removal_key_normalized,
                replacement,
            ];

            let expected_vec = vec![
                load_expected_f32::<TestBackend, 3>("block_0_att_output_v_first", device),
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_r", device),
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_w", device),
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_k", device),
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_v", device),
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_z", device),
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_b", device),
            ];

            let module_names = vec![
                "weight_prepare_v_first".to_string(),
                "weight_prepare_r".to_string(),
                "weight_prepare_w".to_string(),
                "weight_prepare_k".to_string(),
                "weight_prepare_v".to_string(),
                "weight_prepare_z".to_string(),
                "weight_prepare_b".to_string(),
            ];

            let weight_prepare_pass = check_closeness_multi(
                actual_vec,
                expected_vec,
                module_names,
                MIN_PASS_RATE,
                RELATIVE_ERROR,
            );
            if !weight_prepare_pass {
                all_checks_passed = false;
                // lora 还要测
            }

            let receptance =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_r", device);
            let weight_decay =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_w", device);
            let replacement_key =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_k", device);
            let value =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_v", device);
            let removal_key_normalized =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_z", device);
            let replacement =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_input_b", device);

            let wkv_receptance_input: Tensor<TestBackend, 4> = receptance.reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_NUM_HEADS,
                TEST_HEAD_SIZE,
            ]);

            let wkv_weight_decay_input: Tensor<TestBackend, 4> = weight_decay.reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_NUM_HEADS,
                TEST_HEAD_SIZE,
            ]);

            let wkv_key_input: Tensor<TestBackend, 4> = replacement_key.reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_NUM_HEADS,
                TEST_HEAD_SIZE,
            ]);

            let wkv_value_input: Tensor<TestBackend, 4> = value.reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_NUM_HEADS,
                TEST_HEAD_SIZE,
            ]);

            let wkv_removal_input: Tensor<TestBackend, 4> = removal_key_normalized.reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_NUM_HEADS,
                TEST_HEAD_SIZE,
            ]);

            let wkv_replacement_input: Tensor<TestBackend, 4> = replacement.reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_NUM_HEADS,
                TEST_HEAD_SIZE,
            ]);

            let (_final_state, _state_accum, wkv7_kernel_output) = wkv7_forward(
                wkv_weight_decay_input,
                wkv_receptance_input,
                wkv_key_input,
                wkv_value_input,
                wkv_removal_input,
                wkv_replacement_input,
                None,
                16,
            );

            let output = wkv7_kernel_output.clone().reshape([
                TEST_BATCH_SIZE,
                TEST_CONTEXT_LENGTH,
                TEST_EMBEDDED_DIM,
            ]);

            let expected_output =
                load_expected_f32::<TestBackend, 3>("block_0_att_wkv7_kernel_output_x", device);

            let wkv_kernel_pass = check_closeness(
                &output,
                &expected_output,
                "wkv7_kernel",
                MIN_PASS_RATE,
                RELATIVE_ERROR,
            );
            if !wkv_kernel_pass {
                all_checks_passed = false;
            }
        }

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
                Tensor::<TestBackend, 2>::zeros([1, TEST_EMBEDDED_DIM], device),
            );

            let expected_channel_mix_output_x = load_expected_f32::<TestBackend, 3>(
                format!("block_{}_ffn_output_x", cell_id).as_str(),
                device,
            );

            channel_mix_outputs.push(channel_mix_output_x);
            expected_channel_mix_outputs.push(expected_channel_mix_output_x);
            module_names.push(format!("cell_{}_channel_mix", cell_id));
        }

        let channel_mix_pass = check_closeness_multi(
            channel_mix_outputs,
            expected_channel_mix_outputs,
            module_names,
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
        if !channel_mix_pass {
            all_checks_passed = false;
        }
    }

    assert!(
        all_checks_passed,
        "auto_regressive_model forward checks failed"
    );
}

#[test]
fn test_rwkv7_0p1b_backward() {
    let device = &get_test_device::<TestBackend>();
    // let model = get_test_model::<TestBackend>(device);
    let mut all_checks_passed = true;

    let weight_decay =
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_w", device);

    let receptance =
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_q", device);

    let key = load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_k", device);

    let value = load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_v", device);

    let removal =
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_z", device);

    let replacement =
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_b", device);

    let state = load_expected_f32::<TestAutodiffBackend, 5>("wkv7_kernel_backward_saved_s", device);

    let removal_state =
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_saved_sa", device);

    let output_grad =
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_input_dy", device);

    let chunk_len = 16;

    let (
        weight_decay_grad,
        receptance_grad,
        key_grad,
        value_grad,
        removal_grad,
        replacement_grad,
        _initial_state_grad,
    ) = TestAutodiffBackend::wkv7_backward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        state.into_primitive().tensor(),
        removal_state.into_primitive().tensor(),
        output_grad.into_primitive().tensor(),
        chunk_len,
    );

    let actual_vec: Vec<Tensor<TestAutodiffBackend, 4>> = vec![
        weight_decay_grad,
        receptance_grad,
        key_grad,
        value_grad,
        removal_grad,
        replacement_grad,
    ]
    .iter()
    .map(|x| Tensor::<TestAutodiffBackend, 4>::from_primitive(TensorPrimitive::Float(x.clone())))
    .collect();

    let expected_vec = vec![
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_dw", device),
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_dq", device),
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_dk", device),
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_dv", device),
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_dz", device),
        load_expected_f32::<TestAutodiffBackend, 4>("wkv7_kernel_backward_db", device),
    ];

    let module_names = vec![
        "wkv7_kernel_backward_dw".to_string(),
        "wkv7_kernel_backward_dq".to_string(),
        "wkv7_kernel_backward_dk".to_string(),
        "wkv7_kernel_backward_dv".to_string(),
        "wkv7_kernel_backward_da".to_string(),
        "wkv7_kernel_backward_db".to_string(),
    ];

    let backward_pass = check_closeness_multi(
        actual_vec,
        expected_vec,
        module_names,
        MIN_PASS_RATE,
        RELATIVE_ERROR,
    );
    if !backward_pass {
        all_checks_passed = false;
    }

    assert!(
        all_checks_passed,
        "auto_regressive_model backward checks failed"
    );
}
