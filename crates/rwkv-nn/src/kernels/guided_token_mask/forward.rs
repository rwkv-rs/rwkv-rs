use burn::tensor::{
    DType,
    Shape,
    ops::{FloatTensor, IntTensor},
};
use burn_cubecl::{CubeElement, CubeRuntime};

use crate::kernels::{
    backend::{BoolElement, CubeBackend, FloatElement, IntElement},
    guided_token_mask::{GuidedTokenMaskBackend, host::apply_guided_token_masks_launch},
};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> GuidedTokenMaskBackend
    for CubeBackend<R, F, I, BT>
where
    F: CubeElement,
{
    fn apply_guided_token_masks(
        logits: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        guided_token_masks: IntTensor<Self>,
        guided_token_mask_words: usize,
    ) -> FloatTensor<Self> {
        apply_guided_token_masks_launch::<R, F, I, BT>(
            logits,
            batch_ids,
            guided_token_masks,
            guided_token_mask_words,
        )
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn_fusion::{
        Fusion,
        FusionBackend,
        FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;

    impl<B: FusionBackend + GuidedTokenMaskBackend> GuidedTokenMaskBackend for Fusion<B> {
        fn apply_guided_token_masks(
            logits: FloatTensor<Self>,
            batch_ids: IntTensor<Self>,
            guided_token_masks: IntTensor<Self>,
            guided_token_mask_words: usize,
        ) -> FloatTensor<Self> {
            let client = logits.client.clone();
            let [active_batch_size, vocab_size] = logits.shape.dims();

            #[derive(Clone, Debug)]
            struct GuidedTokenMaskOp<B1> {
                desc: CustomOpIr,
                guided_token_mask_words: usize,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + GuidedTokenMaskBackend> Operation<B1::FusionRuntime>
                for GuidedTokenMaskOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([logits, batch_ids, guided_token_masks], [output_out]) =
                        self.desc.as_fixed();

                    let logits_tensor = handles.get_float_tensor::<B1>(logits);
                    let batch_ids_tensor = handles.get_int_tensor::<B1>(batch_ids);
                    let guided_token_masks_tensor =
                        handles.get_int_tensor::<B1>(guided_token_masks);
                    let output = B1::apply_guided_token_masks(
                        logits_tensor,
                        batch_ids_tensor,
                        guided_token_masks_tensor,
                        self.guided_token_mask_words,
                    );

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&logits);
            streams.tensor(&batch_ids);
            streams.tensor(&guided_token_masks);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([active_batch_size, vocab_size]),
                DType::F32,
            )];

            let desc = CustomOpIr::new(
                "guided_token_mask",
                &[
                    logits.into_ir(),
                    batch_ids.into_ir(),
                    guided_token_masks.into_ir(),
                ],
                &output_desc,
            );

            let op = GuidedTokenMaskOp::<B> {
                desc,
                guided_token_mask_words,
                _b: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing guided token mask output")
        }
    }
}
