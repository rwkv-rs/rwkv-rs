use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
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
        weight_decay: burn::tensor::ops::FloatTensor<Self>,
        receptance: burn::tensor::ops::FloatTensor<Self>,
        key: burn::tensor::ops::FloatTensor<Self>,
        value: burn::tensor::ops::FloatTensor<Self>,
        removal: burn::tensor::ops::FloatTensor<Self>,
        replacement: burn::tensor::ops::FloatTensor<Self>,
        initial_state: burn::tensor::ops::FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7ForwardOutput<burn::tensor::ops::FloatTensor<Self>> {
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
        weight_decay: burn::tensor::ops::FloatTensor<Self>,
        receptance: burn::tensor::ops::FloatTensor<Self>,
        key: burn::tensor::ops::FloatTensor<Self>,
        value: burn::tensor::ops::FloatTensor<Self>,
        removal: burn::tensor::ops::FloatTensor<Self>,
        replacement: burn::tensor::ops::FloatTensor<Self>,
        state: burn::tensor::ops::FloatTensor<Self>,
        removal_state: burn::tensor::ops::FloatTensor<Self>,
        output_grad: burn::tensor::ops::FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7StateBackwardOutput<burn::tensor::ops::FloatTensor<Self>> {
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
