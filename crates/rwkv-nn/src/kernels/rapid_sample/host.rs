use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
use burn::tensor::{
    DType, Shape,
    ops::{FloatTensor, IntTensor},
};
use burn_cubecl::{
    CubeRuntime,
    cubecl::prelude::ScalarArg,
    cubecl::{CubeCount, CubeDim},
    kernel::{cast, into_contiguous},
    ops::numeric::empty_device,
};

use crate::kernels::rapid_sample::{
    RapidSampleOutput, RapidSamplePenaltyConfig,
    kernel::{
        RapidSampleConfig, RapidSampleRepetitionInputsLaunch, RapidSampleRepetitionOutputsLaunch,
        RapidSampleTemperatureInputsLaunch, RapidSampleTemperatureOutputsLaunch,
        rapid_sample_repetition_temperature_topk_topp_kernel,
        rapid_sample_temperature_topk_topp_kernel,
    },
};

const MIN_TEMP: f32 = 0.001;
const MAX_TEMP: f32 = 1000f32;

#[allow(clippy::too_many_arguments)]
pub(crate) fn rapid_sample_topk_topp_impl<
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
>(
    logits: FloatTensor<CubeBackend<R, F, I, BT>>,
    states: IntTensor<CubeBackend<R, F, I, BT>>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    penalties: Option<(
        FloatTensor<CubeBackend<R, F, I, BT>>,
        RapidSamplePenaltyConfig,
    )>,
) -> RapidSampleOutput<FloatTensor<CubeBackend<R, F, I, BT>>, IntTensor<CubeBackend<R, F, I, BT>>> {
    assert!(
        temperature >= MIN_TEMP && temperature <= MAX_TEMP,
        "temperature must be in [{MIN_TEMP}, {MAX_TEMP}], got {temperature}"
    );

    let logits = into_contiguous(logits);
    let logits = cast::<R>(logits, DType::F32);

    let states = into_contiguous(states);
    let states = cast::<R>(states, DType::U32);

    let client = logits.client.clone();
    let device = logits.device.clone();
    let shape = logits.shape.clone();

    assert_eq!(
        shape.num_dims(),
        2,
        "logits must have shape [batch_size, vocab_size]"
    );

    let batch_size = shape.dims[0];
    let vocab_size = shape.dims[1];

    assert!(batch_size > 0, "batch_size must be > 0");
    assert!(vocab_size > 0, "vocab_size must be > 0");
    assert_eq!(
        vocab_size % 4,
        0,
        "vocab_size must be a multiple of 4 for rapid_sample, got {vocab_size}"
    );

    let expected_states_shape = Shape::new([batch_size]);
    assert_eq!(
        states.shape, expected_states_shape,
        "states must have shape [batch_size]"
    );

    let (top_k, top_p) = normalize_topk_topp(vocab_size, top_k, top_p);
    let inv_temp = 1f32 / temperature;

    let max_units_per_cube = client.properties().hardware.max_units_per_cube as usize;
    let block_size = select_block_size(vocab_size, max_units_per_cube);
    assert!(
        vocab_size <= block_size * block_size,
        "vocab_size too large for block_size={block_size} (max={}): vocab_size={vocab_size}",
        block_size * block_size
    );

    let config = RapidSampleConfig {
        block_size,
        num_warps: block_size / 32,
    };

    let token_ids =
        empty_device::<R, i32>(client.clone(), device.clone(), Shape::new([batch_size]));
    let probs = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, vocab_size]),
    );

    let cube_count = CubeCount::Static(batch_size as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(block_size as u32);

    match penalties {
        None => {
            rapid_sample_temperature_topk_topp_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                RapidSampleTemperatureInputsLaunch::new(logits.as_tensor_arg(4)),
                RapidSampleTemperatureOutputsLaunch::new(
                    token_ids.as_tensor_arg(1),
                    states.as_tensor_arg(1),
                    probs.as_tensor_arg(4),
                ),
                ScalarArg::new(vocab_size as u32),
                ScalarArg::new(inv_temp),
                ScalarArg::new(top_k),
                ScalarArg::new(top_p),
                config,
            )
            .expect("rapid_sample_temperature_topk_topp_kernel should never fail");

            RapidSampleOutput {
                token_ids,
                states,
                penalties: None,
            }
        }
        Some((penalties, penalty_cfg)) => {
            let penalties = into_contiguous(penalties);
            let penalties = cast::<R>(penalties, DType::F32);

            assert_eq!(
                penalties.shape, shape,
                "penalties must have the same shape as logits"
            );

            rapid_sample_repetition_temperature_topk_topp_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                RapidSampleRepetitionInputsLaunch::new(logits.as_tensor_arg(4)),
                RapidSampleRepetitionOutputsLaunch::new(
                    token_ids.as_tensor_arg(1),
                    penalties.as_tensor_arg(4),
                    states.as_tensor_arg(1),
                    probs.as_tensor_arg(4),
                ),
                ScalarArg::new(vocab_size as u32),
                ScalarArg::new(penalty_cfg.presence_penalty),
                ScalarArg::new(penalty_cfg.repetition_penalty),
                ScalarArg::new(penalty_cfg.penalty_decay),
                ScalarArg::new(inv_temp),
                ScalarArg::new(top_k),
                ScalarArg::new(top_p),
                config,
            )
            .expect("rapid_sample_repetition_temperature_topk_topp_kernel should never fail");

            RapidSampleOutput {
                token_ids,
                states,
                penalties: Some(penalties),
            }
        }
    }
}

fn normalize_topk_topp(vocab_size: usize, mut top_k: i32, mut top_p: f32) -> (u32, f32) {
    if top_k <= 0 || top_k as usize > vocab_size {
        top_k = vocab_size as i32;
    }

    if !(0f32..=1f32).contains(&top_p) {
        top_p = 1f32;
    }

    if top_p == 0f32 {
        // Match rapid-sampling behavior: deterministic argmax when top_p==0.
        top_k = 1;
        top_p = 1f32;
    }

    (top_k as u32, top_p)
}

fn desired_block_size(vocab_size: usize) -> usize {
    // The rapid-sampling 2-phase mapping supports vocab sizes up to `block_size^2`.
    if vocab_size <= 256 * 256 { 256 } else { 1024 }
}

fn select_block_size(vocab_size: usize, max_units_per_cube: usize) -> usize {
    let desired = desired_block_size(vocab_size);
    if desired <= max_units_per_cube {
        return desired;
    }

    // Try to fallback to 256 if the vocab size allows it.
    if 256 <= max_units_per_cube && vocab_size <= 256 * 256 {
        return 256;
    }

    panic!(
        "rapid_sample: required block_size={desired} exceeds device \
         max_units_per_cube={max_units_per_cube} (vocab_size={vocab_size})"
    );
}
