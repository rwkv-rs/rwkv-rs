use burn::{
    backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement},
    cubecl,
    tensor::{Shape, ops::FloatTensor},
};
use burn_cubecl::{
    CubeElement, CubeRuntime,
    kernel::{cast, into_contiguous},
    ops::numeric::{empty_device, zeros_device},
};
use cubecl::{CubeCount, CubeDim};

use crate::kernels::wkv7::{
    Wkv7Backend,
    kernel::{
        Wkv7BackwardInputsLaunch, Wkv7BackwardOutputsLaunch, Wkv7Config, Wkv7InputsLaunch,
        Wkv7OutputsLaunch, wkv7_backward_kernel, wkv7_forward_kernel,
    },
};

macro_rules! wkv7_with_f32_precision {
    (forward: $self:expr, $weight_decay:expr, $receptance:expr, $key:expr, $value:expr, $removal:expr, $replacement:expr, $initial_state:expr => $chunk_len:expr) => {{
        if size_of::<F>() != size_of::<f32>() {
            let result_f32 = wkv7_forward_impl::<R, f32, I, BT>(
                cast::<R, F, f32>($weight_decay),
                cast::<R, F, f32>($receptance),
                cast::<R, F, f32>($key),
                cast::<R, F, f32>($value),
                cast::<R, F, f32>($removal),
                cast::<R, F, f32>($replacement),
                $initial_state.map(|s| cast::<R, F, f32>(s)),
                $chunk_len,
            );

            (
                cast::<R, f32, F>(result_f32.0),
                cast::<R, f32, F>(result_f32.1),
                cast::<R, f32, F>(result_f32.2),
            )
        } else {
            wkv7_forward_impl::<R, F, I, BT>(
                $weight_decay,
                $receptance,
                $key,
                $value,
                $removal,
                $replacement,
                $initial_state,
                $chunk_len,
            )
        }
    }};

    (backward: $self:expr, $weight_decay:expr, $receptance:expr, $key:expr, $value:expr, $removal:expr, $replacement:expr, $state:expr, $removal_state:expr, $output_grad:expr => $chunk_len:expr) => {{
        if size_of::<F>() != size_of::<f32>() {
            // bf16 -> f32计算 -> bf16
            let result_f32 = wkv7_backward_impl::<R, f32, I, BT>(
                cast::<R, F, f32>($weight_decay),
                cast::<R, F, f32>($receptance),
                cast::<R, F, f32>($key),
                cast::<R, F, f32>($value),
                cast::<R, F, f32>($removal),
                cast::<R, F, f32>($replacement),
                cast::<R, F, f32>($state),
                cast::<R, F, f32>($removal_state),
                cast::<R, F, f32>($output_grad),
                $chunk_len,
            );

            (
                cast::<R, f32, F>(result_f32.0),
                cast::<R, f32, F>(result_f32.1),
                cast::<R, f32, F>(result_f32.2),
                cast::<R, f32, F>(result_f32.3),
                cast::<R, f32, F>(result_f32.4),
                cast::<R, f32, F>(result_f32.5),
                cast::<R, f32, F>(result_f32.6),
            )
        } else {
            wkv7_backward_impl::<R, F, I, BT>(
                $weight_decay,
                $receptance,
                $key,
                $value,
                $removal,
                $replacement,
                $state,
                $removal_state,
                $output_grad,
                $chunk_len,
            )
        }
    }};
}

/// 通用的WKV7前向实现

fn wkv7_forward_impl<R: CubeRuntime, FE: FloatElement, I: IntElement, BT: BoolElement>(
    weight_decay: FloatTensor<CubeBackend<R, FE, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, FE, I, BT>>,
    key: FloatTensor<CubeBackend<R, FE, I, BT>>,
    value: FloatTensor<CubeBackend<R, FE, I, BT>>,
    removal: FloatTensor<CubeBackend<R, FE, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, FE, I, BT>>,
    initial_state: Option<FloatTensor<CubeBackend<R, FE, I, BT>>>,
    chunk_len: usize,
) -> (
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
) {
    let weight_decay = into_contiguous(weight_decay);

    let receptance = into_contiguous(receptance);

    let key = into_contiguous(key);

    let value = into_contiguous(value);

    let removal = into_contiguous(removal);

    let replacement = into_contiguous(replacement);

    let client = weight_decay.client.clone();

    let device = weight_decay.device.clone();

    let shape = weight_decay.shape.clone();

    let batch_size = shape.dims[0];

    let seq_len = shape.dims[1];

    let num_heads = shape.dims[2];

    let dim = shape.dims[3];

    let output = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let sa_out = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let num_chunks = (seq_len + chunk_len - 1) / chunk_len;

    let state_shape = Shape::new([batch_size, num_heads, num_chunks, dim, dim]);

    let states = empty_device::<R, FE>(client.clone(), device.clone(), state_shape);

    let initial_state_shape = Shape::new([batch_size, num_heads, dim, dim]);

    let initial_state = initial_state.map(into_contiguous).unwrap_or_else(|| {
        zeros_device::<R, FE>(client.clone(), device.clone(), initial_state_shape.clone())
    });

    let config = Wkv7Config {
        _batch_size: batch_size as u32,
        sequence_length: seq_len as u32,
        num_heads: num_heads as u32,
        head_size: dim as u32,
        chunk_length: chunk_len as u32,
    };

    let cube_count = CubeCount::Static(num_heads as u32, batch_size as u32, 1);

    let cube_dim = CubeDim::new(dim as u32, 1, 1);

    wkv7_forward_kernel::launch::<FE, R>(
        &client,
        cube_count,
        cube_dim,
        Wkv7InputsLaunch::new(
            weight_decay.as_tensor_arg::<FE>(1),
            receptance.as_tensor_arg::<FE>(1),
            key.as_tensor_arg::<FE>(1),
            value.as_tensor_arg::<FE>(1),
            removal.as_tensor_arg::<FE>(1),
            replacement.as_tensor_arg::<FE>(1),
            initial_state.as_tensor_arg::<FE>(1),
        ),
        Wkv7OutputsLaunch::new(
            states.as_tensor_arg::<FE>(1),
            sa_out.as_tensor_arg::<FE>(1),
            output.as_tensor_arg::<FE>(1),
        ),
        config,
    );

    (states, sa_out, output)
}

/// 通用的WKV7反向实现

fn wkv7_backward_impl<R: CubeRuntime, FE: FloatElement, I: IntElement, BT: BoolElement>(
    weight_decay: FloatTensor<CubeBackend<R, FE, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, FE, I, BT>>,
    key: FloatTensor<CubeBackend<R, FE, I, BT>>,
    value: FloatTensor<CubeBackend<R, FE, I, BT>>,
    removal: FloatTensor<CubeBackend<R, FE, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, FE, I, BT>>,
    state: FloatTensor<CubeBackend<R, FE, I, BT>>,
    removal_state: FloatTensor<CubeBackend<R, FE, I, BT>>,
    output_grad: FloatTensor<CubeBackend<R, FE, I, BT>>,
    chunk_len: usize,
) -> (
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
    FloatTensor<CubeBackend<R, FE, I, BT>>,
) {
    let weight_decay = into_contiguous(weight_decay);

    let receptance = into_contiguous(receptance);

    let key = into_contiguous(key);

    let value = into_contiguous(value);

    let removal = into_contiguous(removal);

    let replacement = into_contiguous(replacement);

    let state = into_contiguous(state);

    let removal_state = into_contiguous(removal_state);

    let output_grad = into_contiguous(output_grad);

    let client = weight_decay.client.clone();

    let device = weight_decay.device.clone();

    let shape = weight_decay.shape.clone();

    let batch_size = shape.dims[0];

    let seq_len = shape.dims[1];

    let num_heads = shape.dims[2];

    let dim = shape.dims[3];

    let weight_decay_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let receptance_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let key_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let value_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let removal_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let replacement_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let initial_state_shape = Shape::new([batch_size, num_heads, dim, dim]);

    let grad_initial_state =
        empty_device::<R, FE>(client.clone(), device.clone(), initial_state_shape);

    let config = Wkv7Config {
        _batch_size: batch_size as u32,
        sequence_length: seq_len as u32,
        num_heads: num_heads as u32,
        head_size: dim as u32,
        chunk_length: chunk_len as u32,
    };

    let cube_count = CubeCount::Static(num_heads as u32, batch_size as u32, 1);

    let cube_dim = CubeDim::new(dim as u32, 1, 1);

    wkv7_backward_kernel::launch::<FE, R>(
        &client,
        cube_count,
        cube_dim,
        Wkv7BackwardInputsLaunch::new(
            weight_decay.as_tensor_arg::<FE>(1),
            receptance.as_tensor_arg::<FE>(1),
            key.as_tensor_arg::<FE>(1),
            value.as_tensor_arg::<FE>(1),
            removal.as_tensor_arg::<FE>(1),
            replacement.as_tensor_arg::<FE>(1),
            state.as_tensor_arg::<FE>(1),
            removal_state.as_tensor_arg::<FE>(1),
            output_grad.as_tensor_arg::<FE>(1),
        ),
        Wkv7BackwardOutputsLaunch::new(
            weight_decay_grad.as_tensor_arg::<FE>(1),
            receptance_grad.as_tensor_arg::<FE>(1),
            key_grad.as_tensor_arg::<FE>(1),
            value_grad.as_tensor_arg::<FE>(1),
            removal_grad.as_tensor_arg::<FE>(1),
            replacement_grad.as_tensor_arg::<FE>(1),
            grad_initial_state.as_tensor_arg::<FE>(1),
        ),
        config,
    );

    (
        weight_decay_grad,
        receptance_grad,
        key_grad,
        value_grad,
        removal_grad,
        replacement_grad,
        grad_initial_state,
    )
}

/// CubeBackend implementation of Wkv7Backend

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Wkv7Backend
    for CubeBackend<R, F, I, BT>
where
    f32: CubeElement,
    F: CubeElement,
{
    fn wkv7_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: Option<FloatTensor<Self>>,
        chunk_len: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        wkv7_with_f32_precision!(
            forward: self,
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            initial_state => chunk_len
        )
    }

    fn wkv7_backward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        state: FloatTensor<Self>,
        removal_state: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        chunk_len: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        wkv7_with_f32_precision!(
            backward: self,
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            state,
            removal_state,
            output_grad => chunk_len
        )
    }
}

/// Fusion backend support - based on burn-vision implementation pattern
#[cfg(feature = "fusion")]

mod fusion_impl {

    use burn::tensor::{Element, Shape};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr};

    use super::*;
    use crate::kernels::wkv7::Wkv7Backend;

    impl<B: FusionBackend + Wkv7Backend> Wkv7Backend for Fusion<B> {
        fn wkv7_forward(
            weight_decay: FloatTensor<Self>,
            receptance: FloatTensor<Self>,
            key: FloatTensor<Self>,
            value: FloatTensor<Self>,
            removal: FloatTensor<Self>,
            replacement: FloatTensor<Self>,
            _initial_state: Option<FloatTensor<Self>>, // Not supported in fusion for now
            chunk_len: usize,
        ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
            let client = weight_decay.client.clone();

            let batch = weight_decay.shape[0];

            let seq_len = weight_decay.shape[1];

            let num_heads = weight_decay.shape[2];

            let dim = weight_decay.shape[3];

            let num_chunks = (seq_len + chunk_len - 1) / chunk_len;

            #[derive(Clone, Debug)]

            struct Wkv7ForwardOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7Backend> Operation<B1::FusionRuntime> for Wkv7ForwardOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [weight_decay, receptance, key, value, removal, replacement],
                        [state_out, removal_state_out, output_out],
                    ) = self.desc.as_fixed();

                    let weight_decay_tensor = handles.get_float_tensor::<B1>(weight_decay);

                    let receptance_tensor = handles.get_float_tensor::<B1>(receptance);

                    let key_tensor = handles.get_float_tensor::<B1>(key);

                    let value_tensor = handles.get_float_tensor::<B1>(value);

                    let removal_tensor = handles.get_float_tensor::<B1>(removal);

                    let replacement_tensor = handles.get_float_tensor::<B1>(replacement);

                    let (state, removal_state, output) = B1::wkv7_forward(
                        weight_decay_tensor,
                        receptance_tensor,
                        key_tensor,
                        value_tensor,
                        removal_tensor,
                        replacement_tensor,
                        None, // initial_state not supported in fusion
                        self.chunk_len,
                    );

                    handles.register_float_tensor::<B1>(&state_out.id, state);

                    handles.register_float_tensor::<B1>(&removal_state_out.id, removal_state);

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();

            streams.tensor(&weight_decay);

            streams.tensor(&receptance);

            streams.tensor(&key);

            streams.tensor(&value);

            streams.tensor(&removal);

            streams.tensor(&replacement);

            let state_out = client.tensor_uninitialized(
                vec![batch, num_heads, num_chunks, dim, dim].into(),
                B::FloatElem::dtype(),
            );

            let removal_state_out = client.tensor_uninitialized(
                vec![batch, seq_len, num_heads, dim].into(),
                B::FloatElem::dtype(),
            );

            let output_out = client.tensor_uninitialized(
                vec![batch, seq_len, num_heads, dim].into(),
                B::FloatElem::dtype(),
            );

            let desc = CustomOpIr::new(
                "wkv7_forward",
                &[
                    weight_decay.into_ir(),
                    receptance.into_ir(),
                    key.into_ir(),
                    value.into_ir(),
                    removal.into_ir(),
                    replacement.into_ir(),
                ],
                &[
                    state_out.to_ir_out(),
                    removal_state_out.to_ir_out(),
                    output_out.to_ir_out(),
                ],
            );

            client.register(
                streams,
                OperationIr::Custom(desc.clone()),
                Wkv7ForwardOp::<B> {
                    desc,
                    chunk_len,
                    _b: core::marker::PhantomData,
                },
            );

            (state_out, removal_state_out, output_out)
        }

        fn wkv7_backward(
            weight_decay: FloatTensor<Self>,
            receptance: FloatTensor<Self>,
            key: FloatTensor<Self>,
            value: FloatTensor<Self>,
            removal: FloatTensor<Self>,
            replacement: FloatTensor<Self>,
            state: FloatTensor<Self>,
            removal_state: FloatTensor<Self>,
            output_grad: FloatTensor<Self>,
            chunk_len: usize,
        ) -> (
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
        ) {
            let client = weight_decay.client.clone();

            let shape = weight_decay.shape.clone();

            #[derive(Clone, Debug)]

            struct Wkv7BackwardOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7Backend> Operation<B1::FusionRuntime> for Wkv7BackwardOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            weight_decay,
                            receptance,
                            key,
                            value,
                            removal,
                            replacement,
                            state,
                            removal_state,
                            output_grad,
                        ],
                        [
                            grad_weight_decay,
                            grad_receptance,
                            grad_key,
                            grad_value,
                            grad_removal,
                            grad_replacement,
                            grad_initial_state,
                        ],
                    ) = self.desc.as_fixed();

                    let weight_decay_tensor = handles.get_float_tensor::<B1>(weight_decay);

                    let receptance_tensor = handles.get_float_tensor::<B1>(receptance);

                    let key_tensor = handles.get_float_tensor::<B1>(key);

                    let value_tensor = handles.get_float_tensor::<B1>(value);

                    let removal_tensor = handles.get_float_tensor::<B1>(removal);

                    let replacement_tensor = handles.get_float_tensor::<B1>(replacement);

                    let state_tensor = handles.get_float_tensor::<B1>(state);

                    let removal_state_tensor = handles.get_float_tensor::<B1>(removal_state);

                    let output_grad_tensor = handles.get_float_tensor::<B1>(output_grad);

                    let (
                        weight_decay_grad,
                        receptance_grad,
                        key_grad,
                        value_grad,
                        removal_grad,
                        replacement_grad,
                        initial_state_grad,
                    ) = B1::wkv7_backward(
                        weight_decay_tensor,
                        receptance_tensor,
                        key_tensor,
                        value_tensor,
                        removal_tensor,
                        replacement_tensor,
                        state_tensor,
                        removal_state_tensor,
                        output_grad_tensor,
                        self.chunk_len,
                    );

                    handles.register_float_tensor::<B1>(&grad_weight_decay.id, weight_decay_grad);

                    handles.register_float_tensor::<B1>(&grad_receptance.id, receptance_grad);

                    handles.register_float_tensor::<B1>(&grad_key.id, key_grad);

                    handles.register_float_tensor::<B1>(&grad_value.id, value_grad);

                    handles.register_float_tensor::<B1>(&grad_removal.id, removal_grad);

                    handles.register_float_tensor::<B1>(&grad_replacement.id, replacement_grad);

                    handles.register_float_tensor::<B1>(&grad_initial_state.id, initial_state_grad);
                }
            }

            let mut streams = OperationStreams::default();

            streams.tensor(&weight_decay);

            streams.tensor(&receptance);

            streams.tensor(&key);

            streams.tensor(&value);

            streams.tensor(&removal);

            streams.tensor(&replacement);

            streams.tensor(&state);

            streams.tensor(&removal_state);

            streams.tensor(&output_grad);

            let grad_weight_decay =
                client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());

            let grad_receptance = client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());

            let grad_key = client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());

            let grad_value = client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());

            let grad_removal = client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());

            let grad_replacement =
                client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());

            let grad_initial_state = client.tensor_uninitialized(
                Shape::new([shape.dims[0], shape.dims[2], shape.dims[3], shape.dims[3]]),
                B::FloatElem::dtype(),
            );

            let desc = CustomOpIr::new(
                "wkv7_backward",
                &[
                    weight_decay.into_ir(),
                    receptance.into_ir(),
                    key.into_ir(),
                    value.into_ir(),
                    removal.into_ir(),
                    replacement.into_ir(),
                    state.into_ir(),
                    removal_state.into_ir(),
                    output_grad.into_ir(),
                ],
                &[
                    grad_weight_decay.to_ir_out(),
                    grad_receptance.to_ir_out(),
                    grad_key.to_ir_out(),
                    grad_value.to_ir_out(),
                    grad_removal.to_ir_out(),
                    grad_replacement.to_ir_out(),
                    grad_initial_state.to_ir_out(),
                ],
            );

            client.register(
                streams,
                OperationIr::Custom(desc.clone()),
                Wkv7BackwardOp::<B> {
                    desc,
                    chunk_len,
                    _b: core::marker::PhantomData,
                },
            );

            (
                grad_weight_decay,
                grad_receptance,
                grad_key,
                grad_value,
                grad_removal,
                grad_replacement,
                grad_initial_state,
            )
        }
    }
}
