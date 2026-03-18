use burn::tensor::ops::FloatTensor;
use burn_cubecl::{
    CubeElement, CubeRuntime,
    cubecl::{CubeCount, CubeDim},
    kernel::into_contiguous,
    ops::numeric::empty_device,
};

use crate::kernels::addcmul::{
    Addcmul5Output, Addcmul5OutputPrimitive, AddcmulBackend,
    kernel::{addcmul_kernel, addcmul5_kernel},
};
use crate::kernels::backend::{BoolElement, CubeBackend, FloatElement, IntElement};

const BLOCK_SIZE: u32 = 256;

fn addcmul_launch<R: CubeRuntime, F: FloatElement + CubeElement, I: IntElement, BT: BoolElement>(
    base: FloatTensor<CubeBackend<R, F, I, BT>>,
    diff: FloatTensor<CubeBackend<R, F, I, BT>>,
    scale: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let base = into_contiguous(base);
    let diff = into_contiguous(diff);
    let scale = into_contiguous(scale);

    let client = base.client.clone();
    let device = base.device.clone();
    let shape = base.meta.shape().clone();

    debug_assert_eq!(diff.meta.shape(), &shape, "diff shape mismatch with base");
    debug_assert_eq!(scale.meta.shape().num_dims(), 3, "scale must be rank 3");

    let output = empty_device::<R, F>(client.clone(), device, shape.clone());
    let numel = shape.num_elements();
    let cube_count = CubeCount::Static(numel.div_ceil(BLOCK_SIZE as usize) as u32, 1, 1);

    addcmul_kernel::launch::<F, R>(
        &client,
        cube_count,
        CubeDim::new_1d(BLOCK_SIZE),
        base.as_tensor_arg(1),
        diff.as_tensor_arg(1),
        scale.as_tensor_arg(1),
        output.clone().as_tensor_arg(1),
    )
    .expect("addcmul_kernel should never fail");

    output
}

#[allow(clippy::too_many_arguments)]
fn addcmul5_launch<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    base: FloatTensor<CubeBackend<R, F, I, BT>>,
    diff: FloatTensor<CubeBackend<R, F, I, BT>>,
    receptance_scale: FloatTensor<CubeBackend<R, F, I, BT>>,
    weight_decay_scale: FloatTensor<CubeBackend<R, F, I, BT>>,
    key_scale: FloatTensor<CubeBackend<R, F, I, BT>>,
    value_scale: FloatTensor<CubeBackend<R, F, I, BT>>,
    learning_rate_scale: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> Addcmul5OutputPrimitive<CubeBackend<R, F, I, BT>> {
    let base = into_contiguous(base);
    let diff = into_contiguous(diff);
    let receptance_scale = into_contiguous(receptance_scale);
    let weight_decay_scale = into_contiguous(weight_decay_scale);
    let key_scale = into_contiguous(key_scale);
    let value_scale = into_contiguous(value_scale);
    let learning_rate_scale = into_contiguous(learning_rate_scale);

    let client = base.client.clone();
    let device = base.device.clone();
    let shape = base.meta.shape().clone();

    debug_assert_eq!(diff.meta.shape(), &shape, "diff shape mismatch with base");

    let receptance_output = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let weight_decay_output = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let key_output = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let value_output = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let learning_rate_output = empty_device::<R, F>(client.clone(), device, shape.clone());
    let numel = shape.num_elements();
    let cube_count = CubeCount::Static(numel.div_ceil(BLOCK_SIZE as usize) as u32, 1, 1);

    addcmul5_kernel::launch::<F, R>(
        &client,
        cube_count,
        CubeDim::new_1d(BLOCK_SIZE),
        base.as_tensor_arg(1),
        diff.as_tensor_arg(1),
        receptance_scale.as_tensor_arg(1),
        weight_decay_scale.as_tensor_arg(1),
        key_scale.as_tensor_arg(1),
        value_scale.as_tensor_arg(1),
        learning_rate_scale.as_tensor_arg(1),
        receptance_output.clone().as_tensor_arg(1),
        weight_decay_output.clone().as_tensor_arg(1),
        key_output.clone().as_tensor_arg(1),
        value_output.clone().as_tensor_arg(1),
        learning_rate_output.clone().as_tensor_arg(1),
    )
    .expect("addcmul5_kernel should never fail");

    Addcmul5Output {
        receptance_input: receptance_output,
        weight_decay_input: weight_decay_output,
        key_input: key_output,
        value_input: value_output,
        learning_rate_input: learning_rate_output,
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> AddcmulBackend
    for CubeBackend<R, F, I, BT>
where
    F: CubeElement,
{
    fn addcmul(
        base: FloatTensor<Self>,
        diff: FloatTensor<Self>,
        scale: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        addcmul_launch::<R, F, I, BT>(base, diff, scale)
    }

    fn addcmul5(
        base: FloatTensor<Self>,
        diff: FloatTensor<Self>,
        receptance_scale: FloatTensor<Self>,
        weight_decay_scale: FloatTensor<Self>,
        key_scale: FloatTensor<Self>,
        value_scale: FloatTensor<Self>,
        learning_rate_scale: FloatTensor<Self>,
    ) -> Addcmul5OutputPrimitive<Self> {
        addcmul5_launch::<R, F, I, BT>(
            base,
            diff,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
        )
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{Element, Shape};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;

    impl<B: FusionBackend + AddcmulBackend> AddcmulBackend for Fusion<B> {
        fn addcmul(
            base: FloatTensor<Self>,
            diff: FloatTensor<Self>,
            scale: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            let client = base.client.clone();
            let [batch_size, context_length, embedded_dim] = base.shape.dims();

            #[derive(Clone, Debug)]
            struct AddcmulOp<B1> {
                desc: CustomOpIr,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + AddcmulBackend> Operation<B1::FusionRuntime> for AddcmulOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([base, diff, scale], [output_out]) = self.desc.as_fixed();

                    let base_tensor = handles.get_float_tensor::<B1>(base);
                    let diff_tensor = handles.get_float_tensor::<B1>(diff);
                    let scale_tensor = handles.get_float_tensor::<B1>(scale);

                    let output = B1::addcmul(base_tensor, diff_tensor, scale_tensor);
                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&base);
            streams.tensor(&diff);
            streams.tensor(&scale);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_length, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "addcmul",
                &[base.into_ir(), diff.into_ir(), scale.into_ir()],
                &output_desc,
            );

            let op = AddcmulOp::<B> {
                desc,
                _b: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing addcmul output")
        }

        fn addcmul5(
            base: FloatTensor<Self>,
            diff: FloatTensor<Self>,
            receptance_scale: FloatTensor<Self>,
            weight_decay_scale: FloatTensor<Self>,
            key_scale: FloatTensor<Self>,
            value_scale: FloatTensor<Self>,
            learning_rate_scale: FloatTensor<Self>,
        ) -> Addcmul5OutputPrimitive<Self> {
            let client = base.client.clone();
            let [batch_size, context_length, embedded_dim] = base.shape.dims();

            #[derive(Clone, Debug)]
            struct Addcmul5Op<B1> {
                desc: CustomOpIr,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + AddcmulBackend> Operation<B1::FusionRuntime> for Addcmul5Op<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            base,
                            diff,
                            receptance_scale,
                            weight_decay_scale,
                            key_scale,
                            value_scale,
                            learning_rate_scale,
                        ],
                        [
                            receptance_output_out,
                            weight_decay_output_out,
                            key_output_out,
                            value_output_out,
                            learning_rate_output_out,
                        ],
                    ) = self.desc.as_fixed();

                    let base_tensor = handles.get_float_tensor::<B1>(base);
                    let diff_tensor = handles.get_float_tensor::<B1>(diff);
                    let receptance_scale_tensor = handles.get_float_tensor::<B1>(receptance_scale);
                    let weight_decay_scale_tensor =
                        handles.get_float_tensor::<B1>(weight_decay_scale);
                    let key_scale_tensor = handles.get_float_tensor::<B1>(key_scale);
                    let value_scale_tensor = handles.get_float_tensor::<B1>(value_scale);
                    let learning_rate_scale_tensor =
                        handles.get_float_tensor::<B1>(learning_rate_scale);

                    let output = B1::addcmul5(
                        base_tensor,
                        diff_tensor,
                        receptance_scale_tensor,
                        weight_decay_scale_tensor,
                        key_scale_tensor,
                        value_scale_tensor,
                        learning_rate_scale_tensor,
                    );

                    handles.register_float_tensor::<B1>(
                        &receptance_output_out.id,
                        output.receptance_input,
                    );
                    handles.register_float_tensor::<B1>(
                        &weight_decay_output_out.id,
                        output.weight_decay_input,
                    );
                    handles.register_float_tensor::<B1>(&key_output_out.id, output.key_input);
                    handles.register_float_tensor::<B1>(&value_output_out.id, output.value_input);
                    handles.register_float_tensor::<B1>(
                        &learning_rate_output_out.id,
                        output.learning_rate_input,
                    );
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&base);
            streams.tensor(&diff);
            streams.tensor(&receptance_scale);
            streams.tensor(&weight_decay_scale);
            streams.tensor(&key_scale);
            streams.tensor(&value_scale);
            streams.tensor(&learning_rate_scale);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
            ];

            let desc = CustomOpIr::new(
                "addcmul5",
                &[
                    base.into_ir(),
                    diff.into_ir(),
                    receptance_scale.into_ir(),
                    weight_decay_scale.into_ir(),
                    key_scale.into_ir(),
                    value_scale.into_ir(),
                    learning_rate_scale.into_ir(),
                ],
                &output_desc,
            );

            let op = Addcmul5Op::<B> {
                desc,
                _b: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);

            let learning_rate_input = outputs.pop().expect("missing learning_rate_input");
            let value_input = outputs.pop().expect("missing value_input");
            let key_input = outputs.pop().expect("missing key_input");
            let weight_decay_input = outputs.pop().expect("missing weight_decay_input");
            let receptance_input = outputs.pop().expect("missing receptance_input");

            Addcmul5Output {
                receptance_input,
                weight_decay_input,
                key_input,
                value_input,
                learning_rate_input,
            }
        }
    }
}
