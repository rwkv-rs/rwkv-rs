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

#[cfg(test)]
mod tests {
    use burn::{
        backend::Cpu,
        tensor::{DType, Tolerance},
    };

    use super::{GUIDED_MASKED_LOGIT, apply_guided_token_masks};

    type TestBackend = Cpu<f32, i32>;

    #[test]
    fn apply_guided_token_masks_masks_only_disallowed_tokens() {
        let device = Default::default();
        let logits = burn::tensor::Tensor::<TestBackend, 2>::from_data(
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            &device,
        );
        let batch_ids =
            burn::tensor::Tensor::<TestBackend, 1, burn::tensor::Int>::from_data([0, 1], &device);
        let guided_token_masks =
            burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Int>::from_data(
                [[0b0101], [-1i32]],
                &device,
            );

        let output = apply_guided_token_masks(logits, batch_ids, guided_token_masks, 1);
        output.into_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::new(
                vec![
                    1.0,
                    GUIDED_MASKED_LOGIT,
                    3.0,
                    GUIDED_MASKED_LOGIT,
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ],
                [2, 4],
            )
            .convert_dtype(DType::F32),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
    }

    #[test]
    fn apply_guided_token_masks_uses_slot_mapped_batch_ids() {
        let device = Default::default();
        let logits = burn::tensor::Tensor::<TestBackend, 2>::from_data(
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            &device,
        );
        let batch_ids =
            burn::tensor::Tensor::<TestBackend, 1, burn::tensor::Int>::from_data([1, 0], &device);
        let guided_token_masks =
            burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Int>::from_data(
                [[0b0101], [0b1010]],
                &device,
            );

        let output = apply_guided_token_masks(logits, batch_ids, guided_token_masks, 1);
        output.into_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::new(
                vec![
                    GUIDED_MASKED_LOGIT,
                    2.0,
                    GUIDED_MASKED_LOGIT,
                    4.0,
                    10.0,
                    GUIDED_MASKED_LOGIT,
                    30.0,
                    GUIDED_MASKED_LOGIT,
                ],
                [2, 4],
            )
            .convert_dtype(DType::F32),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
    }
}
