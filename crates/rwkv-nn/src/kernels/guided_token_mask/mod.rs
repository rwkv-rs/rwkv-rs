mod forward;
mod host;
mod kernel;

use burn::{
    prelude::Backend,
    tensor::{
        Int,
        Tensor,
        TensorPrimitive,
        ops::{FloatTensor, IntTensor},
    },
};

pub(crate) const GUIDED_MASKED_LOGIT: f32 = -1.0e30;

pub trait GuidedTokenMaskBackend: Backend {
    fn apply_guided_token_masks(
        logits: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        guided_token_masks: IntTensor<Self>,
        guided_token_mask_words: usize,
    ) -> FloatTensor<Self>;
}

pub fn apply_guided_token_masks<B: GuidedTokenMaskBackend>(
    logits: Tensor<B, 2>,
    batch_ids: Tensor<B, 1, Int>,
    guided_token_masks: Tensor<B, 2, Int>,
    guided_token_mask_words: usize,
) -> Tensor<B, 2> {
    let output = B::apply_guided_token_masks(
        logits.into_primitive().tensor(),
        batch_ids.into_primitive(),
        guided_token_masks.into_primitive(),
        guided_token_mask_words,
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
