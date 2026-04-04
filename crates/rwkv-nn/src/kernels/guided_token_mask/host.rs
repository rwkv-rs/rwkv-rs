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
    guided_token_mask::{GUIDED_MASKED_LOGIT, kernel::apply_guided_token_masks_kernel},
};

const BLOCK_SIZE: u32 = 256;

pub(crate) fn apply_guided_token_masks_launch<
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
>(
    logits: FloatTensor<CubeBackend<R, F, I, BT>>,
    batch_ids: IntTensor<CubeBackend<R, F, I, BT>>,
    guided_token_masks: IntTensor<CubeBackend<R, F, I, BT>>,
    guided_token_mask_words: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let logits = cast::<R>(into_contiguous(logits), DType::F32);
    let batch_ids = cast::<R>(into_contiguous(batch_ids), DType::U32);
    let guided_token_masks = cast::<R>(into_contiguous(guided_token_masks), DType::U32);

    let client = logits.client.clone();
    let device = logits.device.clone();
    let logits_shape = logits.meta.shape().clone();
    debug_assert_eq!(
        logits_shape.num_dims(),
        2,
        "guided token masks expect logits with shape [active_batch_size, vocab_size]"
    );

    let active_batch_size = logits_shape[0];
    let vocab_size = logits_shape[1];
    let full_batch_shape = guided_token_masks.meta.shape().clone();
    debug_assert_eq!(
        full_batch_shape.num_dims(),
        2,
        "guided token masks expect state shape [full_batch_size, guided_token_mask_words]"
    );
    let full_batch_size = full_batch_shape[0];
    debug_assert!(vocab_size > 0, "guided token masks expect vocab_size > 0");
    debug_assert_eq!(
        batch_ids.meta.shape(),
        &Shape::new([active_batch_size]),
        "guided token masks expect batch_ids with shape [active_batch_size]"
    );
    debug_assert_eq!(
        guided_token_masks.meta.shape(),
        &Shape::new([full_batch_size, guided_token_mask_words]),
        "guided token masks shape mismatch with full batch"
    );

    let output = empty_device::<R, f32>(client.clone(), device, logits_shape.clone());
    let cube_count = CubeCount::Static(
        logits_shape.num_elements().div_ceil(BLOCK_SIZE as usize) as u32,
        1,
        1,
    );

    apply_guided_token_masks_kernel::launch(
        &client,
        cube_count,
        CubeDim::new_1d(BLOCK_SIZE),
        logits.as_tensor_arg(1),
        batch_ids.as_tensor_arg(1),
        guided_token_masks.as_tensor_arg(1),
        output.clone().as_tensor_arg(1),
        ScalarArg::new(vocab_size as u32),
        ScalarArg::new(guided_token_mask_words as u32),
        ScalarArg::new(GUIDED_MASKED_LOGIT),
    )
    .expect("apply_guided_token_masks_kernel should never fail");

    output
}
