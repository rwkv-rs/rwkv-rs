use burn::tensor::{
    DType,
    Shape,
    ops::{FloatTensor, IntTensor},
};
use burn_cubecl::{
    CubeRuntime,
    cubecl::{CubeCount, CubeDim, prelude::ScalarArg},
    kernel::{cast, into_contiguous},
    ops::numeric::empty_device,
};

use crate::kernels::{
    backend::{BoolElement, CubeBackend, FloatElement, IntElement},
    rapid_sample::{
        RapidSampleOutput,
        kernel::{
            RapidSampleBatchIdsInputsLaunch,
            RapidSampleConfig,
            RapidSamplePenaltyParamsLaunch,
            RapidSampleRepetitionOutputsLaunch,
            rapid_sample_repetition_temperature_topk_topp_batch_ids_kernel,
        },
    },
};

const RAPID_SAMPLE_BLOCK_SIZE: usize = 1024;
const RAPID_SAMPLE_MAX_VOCAB_SIZE: usize = RAPID_SAMPLE_BLOCK_SIZE * RAPID_SAMPLE_BLOCK_SIZE;

#[allow(clippy::too_many_arguments)]
pub(crate) fn rapid_sample_topk_topp_impl<
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
>(
    logits: FloatTensor<CubeBackend<R, F, I, BT>>,
    batch_ids: IntTensor<CubeBackend<R, F, I, BT>>,
    states: IntTensor<CubeBackend<R, F, I, BT>>,
    inv_temperatures: FloatTensor<CubeBackend<R, F, I, BT>>,
    top_ks: IntTensor<CubeBackend<R, F, I, BT>>,
    top_ps: FloatTensor<CubeBackend<R, F, I, BT>>,
    penalties: (
        FloatTensor<CubeBackend<R, F, I, BT>>,
        FloatTensor<CubeBackend<R, F, I, BT>>,
        FloatTensor<CubeBackend<R, F, I, BT>>,
        FloatTensor<CubeBackend<R, F, I, BT>>,
    ),
) -> RapidSampleOutput<FloatTensor<CubeBackend<R, F, I, BT>>, IntTensor<CubeBackend<R, F, I, BT>>> {
    let logits = into_contiguous(logits);
    let logits = cast::<R>(logits, DType::F32);

    let batch_ids = into_contiguous(batch_ids);
    let batch_ids = cast::<R>(batch_ids, DType::U32);

    let states = into_contiguous(states);
    let states = cast::<R>(states, DType::U32);

    let inv_temperatures = into_contiguous(inv_temperatures);
    let inv_temperatures = cast::<R>(inv_temperatures, DType::F32);

    let top_ks = into_contiguous(top_ks);
    let top_ks = cast::<R>(top_ks, DType::U32);

    let top_ps = into_contiguous(top_ps);
    let top_ps = cast::<R>(top_ps, DType::F32);

    let (penalties, presence_penalty, repetition_penalty, penalty_decay) = penalties;
    let penalties = into_contiguous(penalties);
    debug_assert_eq!(
        penalties.dtype,
        DType::F32,
        "rapid_sample_batch_ids: penalties must be F32, got {:?}",
        penalties.dtype
    );

    let presence_penalty = into_contiguous(presence_penalty);
    let presence_penalty = cast::<R>(presence_penalty, DType::F32);
    let repetition_penalty = into_contiguous(repetition_penalty);
    let repetition_penalty = cast::<R>(repetition_penalty, DType::F32);
    let penalty_decay = into_contiguous(penalty_decay);
    let penalty_decay = cast::<R>(penalty_decay, DType::F32);

    let client = logits.client.clone();
    let device = logits.device.clone();
    let logits_shape = logits.meta.shape().clone();
    debug_assert_eq!(
        logits_shape.num_dims(),
        2,
        "logits must have shape [active_batch_size, vocab_size]"
    );

    let active_batch_size = logits_shape[0];
    let vocab_size = logits_shape[1];
    let full_batch_shape = states.meta.shape().clone();
    debug_assert_eq!(
        full_batch_shape.num_dims(),
        1,
        "states must have shape [full_batch_size]"
    );
    let full_batch_size = full_batch_shape[0];

    debug_assert!(active_batch_size > 0, "active_batch_size must be > 0");
    debug_assert!(full_batch_size > 0, "full_batch_size must be > 0");
    debug_assert!(vocab_size > 0, "vocab_size must be > 0");
    debug_assert_eq!(
        vocab_size % 4,
        0,
        "vocab_size must be a multiple of 4 for rapid_sample, got {vocab_size}"
    );

    let expected_active_1d = Shape::new([active_batch_size]);
    debug_assert_eq!(
        batch_ids.meta.shape(),
        &expected_active_1d,
        "batch_ids must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        inv_temperatures.meta.shape(),
        &expected_active_1d,
        "inv_temperatures must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        top_ks.meta.shape(),
        &expected_active_1d,
        "top_ks must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        top_ps.meta.shape(),
        &expected_active_1d,
        "top_ps must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        presence_penalty.meta.shape(),
        &expected_active_1d,
        "presence_penalty must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        repetition_penalty.meta.shape(),
        &expected_active_1d,
        "repetition_penalty must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        penalty_decay.meta.shape(),
        &expected_active_1d,
        "penalty_decay must have shape [active_batch_size]"
    );
    debug_assert_eq!(
        penalties.meta.shape(),
        &Shape::new([full_batch_size, vocab_size]),
        "penalties must have shape [full_batch_size, vocab_size]"
    );

    let max_units_per_cube = client.properties().hardware.max_units_per_cube as usize;
    let block_size = select_block_size(vocab_size, max_units_per_cube);
    debug_assert!(
        vocab_size <= block_size * block_size,
        "vocab_size too large for block_size={block_size} (max={}): vocab_size={vocab_size}",
        block_size * block_size
    );

    let config = RapidSampleConfig {
        block_size,
        num_warps: block_size / 32,
    };

    let token_ids = empty_device::<R, i32>(
        client.clone(),
        device.clone(),
        Shape::new([active_batch_size]),
    );
    let probs = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([active_batch_size, vocab_size]),
    );

    let cube_count = CubeCount::Static(active_batch_size as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(block_size as u32);

    rapid_sample_repetition_temperature_topk_topp_batch_ids_kernel::launch(
        &client,
        cube_count,
        cube_dim,
        RapidSampleBatchIdsInputsLaunch::new(logits.as_tensor_arg(4), batch_ids.as_tensor_arg(1)),
        RapidSampleRepetitionOutputsLaunch::new(
            token_ids.as_tensor_arg(1),
            penalties.as_tensor_arg(4),
            states.as_tensor_arg(1),
            probs.as_tensor_arg(4),
        ),
        RapidSamplePenaltyParamsLaunch::new(
            inv_temperatures.as_tensor_arg(1),
            top_ks.as_tensor_arg(1),
            top_ps.as_tensor_arg(1),
            presence_penalty.as_tensor_arg(1),
            repetition_penalty.as_tensor_arg(1),
            penalty_decay.as_tensor_arg(1),
        ),
        ScalarArg::new(vocab_size as u32),
        config,
    )
    .expect("rapid_sample_repetition_temperature_topk_topp_batch_ids_kernel should never fail");

    RapidSampleOutput {
        token_ids,
        states,
        probs,
        penalties: Some(penalties),
    }
}

fn select_block_size(vocab_size: usize, max_units_per_cube: usize) -> usize {
    // Match the reference CUDA implementation directly: it is tuned around a fixed
    // `BLOCKDIM_X_SAMPLE == 1024` launch and the tile-selection math relies on the resulting
    // `block_size^2` coverage. The profiling pass confirmed this CubeCL port stays within the
    // intended <=64 regs/thread budget at 1024 threads, so the default path should no longer
    // downshift to smaller blocks.
    assert!(
        RAPID_SAMPLE_BLOCK_SIZE <= max_units_per_cube,
        "rapid_sample: reference block_size={} exceeds device max_units_per_cube={max_units_per_cube}",
        RAPID_SAMPLE_BLOCK_SIZE
    );
    assert!(
        vocab_size <= RAPID_SAMPLE_MAX_VOCAB_SIZE,
        "rapid_sample: vocab_size={vocab_size} exceeds fixed 1024-thread coverage max={}",
        RAPID_SAMPLE_MAX_VOCAB_SIZE
    );

    RAPID_SAMPLE_BLOCK_SIZE
}
