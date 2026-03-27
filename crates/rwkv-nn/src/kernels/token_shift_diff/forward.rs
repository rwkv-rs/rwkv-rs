use burn::tensor::ops::FloatTensor;
use burn_cubecl::{
    CubeElement,
    CubeRuntime,
    cubecl::{CubeCount, CubeDim},
    kernel::into_contiguous,
    ops::numeric::empty_device,
};

use crate::kernels::{
    backend::{BoolElement, CubeBackend, FloatElement, IntElement},
    token_shift_diff::{TokenShiftDiffBackend, kernel::rwkv_token_shift_diff_kernel},
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
    context_mask: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let embedded_context = into_contiguous(embedded_context);
    let embedded_token_shift = into_contiguous(embedded_token_shift);
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
        context_mask.meta.shape().num_dims(),
        2,
        "context_mask must be rank 2"
    );

    let output = empty_device::<R, F>(client.clone(), device, shape.clone());
    let numel = shape.num_elements();
    let cube_count = CubeCount::Static(numel.div_ceil(BLOCK_SIZE as usize) as u32, 1, 1);

    rwkv_token_shift_diff_kernel::launch::<F, R>(
        &client,
        cube_count,
        CubeDim::new_1d(BLOCK_SIZE),
        embedded_context.as_tensor_arg(1),
        embedded_token_shift.as_tensor_arg(1),
        context_mask.as_tensor_arg(1),
        output.clone().as_tensor_arg(1),
    )
    .expect("rwkv_token_shift_diff_kernel should never fail");

    output
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> TokenShiftDiffBackend
    for CubeBackend<R, F, I, BT>
where
    F: CubeElement,
{
    fn token_shift_diff(
        embedded_context: FloatTensor<Self>,
        embedded_token_shift: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        rwkv_token_shift_diff_launch::<R, F, I, BT>(
            embedded_context,
            embedded_token_shift,
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
            context_mask: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            let client = embedded_context.client.clone();
            let [batch_size, context_length, embedded_dim] = embedded_context.shape.dims();

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
                    let ([embedded_context, embedded_token_shift, context_mask], [output_out]) =
                        self.desc.as_fixed();

                    let embedded_context_tensor = handles.get_float_tensor::<B1>(embedded_context);
                    let embedded_token_shift_tensor =
                        handles.get_float_tensor::<B1>(embedded_token_shift);
                    let context_mask_tensor = handles.get_float_tensor::<B1>(context_mask);

                    let output = B1::token_shift_diff(
                        embedded_context_tensor,
                        embedded_token_shift_tensor,
                        context_mask_tensor,
                    );

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&embedded_context);
            streams.tensor(&embedded_token_shift);
            streams.tensor(&context_mask);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_length, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "token_shift_diff",
                &[
                    embedded_context.into_ir(),
                    embedded_token_shift.into_ir(),
                    context_mask.into_ir(),
                ],
                &output_desc,
            );

            let op = TokenShiftDiffOp::<B> {
                desc,
                _b: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing token_shift_diff output")
        }
    }
}
