use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
use burn::tensor::ops::FloatTensor;
use burn_cubecl::{CubeElement, CubeRuntime};

use crate::kernels::wkv7_common::{
    Wkv7ForwardOutput, Wkv7StateBackwardOutput,
    host::{wkv7_backward_impl, wkv7_forward_impl},
};
use crate::kernels::wkv7_statetune::Wkv7StateTuneBackend;

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Wkv7StateTuneBackend
    for CubeBackend<R, F, I, BT>
where
    f32: CubeElement,
    F: CubeElement,
{
    fn wkv7_statetune_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7ForwardOutput<FloatTensor<Self>> {
        let output = wkv7_forward_impl::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            Some(initial_state),
            chunk_len,
            true,
            false,
        );

        Wkv7ForwardOutput {
            state: output.state,
            removal_state: output.removal_state,
            output: output.output,
        }
    }

    fn wkv7_statetune_backward(
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
    ) -> Wkv7StateBackwardOutput<FloatTensor<Self>> {
        wkv7_backward_impl::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            state,
            removal_state,
            output_grad,
            None,
            chunk_len,
            false,
            true,
        )
    }
}
