use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::{
    CubeElement,
    CubeRuntime,
    cubecl::{CubeCount, CubeDim},
    kernel::{cast, into_contiguous},
    ops::numeric::empty_device,
};

use crate::kernels::{
    backend::{BoolElement, CubeBackend, FloatElement, IntElement},
    token_shift_diff::{
        TokenShiftDiffBackend,
        TokenShiftDiffPrimitiveOutput,
        kernel::{
            TokenShiftDiffInputsLaunch,
            TokenShiftDiffOutputsLaunch,
            rwkv_token_shift_diff_kernel,
        },
    },
};

const BLOCK_SIZE: u32 = 256;

fn rwkv_token_shift_diff_launch<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    embedded_context: FloatTensor<CubeBackend<R, F, I, BT>>,
    embedded_token_shift: FloatTensor<CubeBackend<R, F, I, BT>>,
    batch_ids: IntTensor<CubeBackend<R, F, I, BT>>,
    context_mask: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> TokenShiftDiffPrimitiveOutput<CubeBackend<R, F, I, BT>> {
    let embedded_context = into_contiguous(embedded_context);
    let embedded_token_shift = into_contiguous(embedded_token_shift);
    let batch_ids = cast::<R>(into_contiguous(batch_ids), burn::tensor::DType::U32);
    let context_mask = into_contiguous(context_mask);

    let client = embedded_context.client.clone();
    let device = embedded_context.device.clone();
    let shape = embedded_context.meta.shape().clone();

    debug_assert_eq!(
        embedded_token_shift.meta.shape().num_dims(),
        2,
        "embedded_token_shift must be rank 2"
    );
    debug_assert_eq!(
        batch_ids.meta.shape().num_dims(),
        1,
        "batch_ids must be rank 1"
    );
    debug_assert_eq!(
        context_mask.meta.shape().num_dims(),
        2,
        "context_mask must be rank 2"
    );
    debug_assert_eq!(
        batch_ids.meta.shape()[0],
        shape[0],
        "batch_ids shape mismatch with embedded_context batch size"
    );
    debug_assert_eq!(
        context_mask.meta.shape()[0],
        shape[0],
        "context_mask batch mismatch with embedded_context batch size"
    );
    debug_assert_eq!(
        context_mask.meta.shape()[1],
        shape[1],
        "context_mask time mismatch with embedded_context"
    );
    debug_assert_eq!(
        embedded_token_shift.meta.shape()[1],
        shape[2],
        "embedded_token_shift feature mismatch with embedded_context"
    );

    let output = empty_device::<R, F>(client.clone(), device, shape.clone());
    let active_feature_count = shape[0] * shape[2];
    let cube_count = CubeCount::Static(
        active_feature_count.div_ceil(BLOCK_SIZE as usize) as u32,
        1,
        1,
    );

    rwkv_token_shift_diff_kernel::launch::<F, R>(
        &client,
        cube_count,
        CubeDim::new_1d(BLOCK_SIZE),
        TokenShiftDiffInputsLaunch::new(
            embedded_context.as_tensor_arg(1),
            batch_ids.as_tensor_arg(1),
            context_mask.as_tensor_arg(1),
        ),
        TokenShiftDiffOutputsLaunch::new(
            output.clone().as_tensor_arg(1),
            embedded_token_shift.as_tensor_arg(1),
        ),
    )
    .expect("rwkv_token_shift_diff_kernel should never fail");

    TokenShiftDiffPrimitiveOutput {
        token_shifted_diff: output,
        next_token_shift: embedded_token_shift,
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> TokenShiftDiffBackend
    for CubeBackend<R, F, I, BT>
where
    F: CubeElement,
{
    fn token_shift_diff(
        embedded_context: FloatTensor<Self>,
        embedded_token_shift: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> TokenShiftDiffPrimitiveOutput<Self> {
        rwkv_token_shift_diff_launch::<R, F, I, BT>(
            embedded_context,
            embedded_token_shift,
            batch_ids,
            context_mask,
        )
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{Element, Shape};
    use burn_fusion::{
        Fusion,
        FusionBackend,
        FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;

    impl<B: FusionBackend + TokenShiftDiffBackend> TokenShiftDiffBackend for Fusion<B> {
        fn token_shift_diff(
            embedded_context: FloatTensor<Self>,
            embedded_token_shift: FloatTensor<Self>,
            batch_ids: IntTensor<Self>,
            context_mask: FloatTensor<Self>,
        ) -> TokenShiftDiffPrimitiveOutput<Self> {
            let client = embedded_context.client.clone();
            let [batch_size, context_length, embedded_dim] = embedded_context.shape.dims();
            let [full_batch_size, _] = embedded_token_shift.shape.dims();

            #[derive(Clone, Debug)]
            struct TokenShiftDiffOp<B1> {
                desc: CustomOpIr,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + TokenShiftDiffBackend> Operation<B1::FusionRuntime>
                for TokenShiftDiffOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            embedded_context,
                            embedded_token_shift,
                            batch_ids,
                            context_mask,
                        ],
                        [token_shifted_diff_out, next_token_shift_out],
                    ) = self.desc.as_fixed();

                    let embedded_context_tensor = handles.get_float_tensor::<B1>(embedded_context);
                    let embedded_token_shift_tensor =
                        handles.get_float_tensor::<B1>(embedded_token_shift);
                    let batch_ids_tensor = handles.get_int_tensor::<B1>(batch_ids);
                    let context_mask_tensor = handles.get_float_tensor::<B1>(context_mask);

                    let output = B1::token_shift_diff(
                        embedded_context_tensor,
                        embedded_token_shift_tensor,
                        batch_ids_tensor,
                        context_mask_tensor,
                    );

                    handles.register_float_tensor::<B1>(
                        &token_shifted_diff_out.id,
                        output.token_shifted_diff,
                    );
                    handles.register_float_tensor::<B1>(
                        &next_token_shift_out.id,
                        output.next_token_shift,
                    );
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&embedded_context);
            streams.tensor(&embedded_token_shift);
            streams.tensor(&batch_ids);
            streams.tensor(&context_mask);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([full_batch_size, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
            ];

            let desc = CustomOpIr::new(
                "token_shift_diff",
                &[
                    embedded_context.into_ir(),
                    embedded_token_shift.into_ir(),
                    batch_ids.into_ir(),
                    context_mask.into_ir(),
                ],
                &output_desc,
            );

            let op = TokenShiftDiffOp::<B> {
                desc,
                _b: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);

            let next_token_shift = outputs.pop().expect("missing next_token_shift output");
            let token_shifted_diff = outputs.pop().expect("missing token_shifted_diff output");

            TokenShiftDiffPrimitiveOutput {
                token_shifted_diff,
                next_token_shift,
            }
        }
    }
}
