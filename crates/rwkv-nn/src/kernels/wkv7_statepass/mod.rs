mod backward;
mod forward;

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, backend::AutodiffBackend, ops::FloatTensor},
};

use crate::kernels::wkv7_common::{Wkv7StateBackwardOutput, Wkv7StatePassForwardOutput};

pub type Wkv7StatePassForwardOutputTensor<B> = Wkv7StatePassForwardOutput<Tensor<B, 4>>;
pub type Wkv7StatePassBackwardOutputTensor<B> = Wkv7StateBackwardOutput<Tensor<B, 4>>;

pub type Wkv7StatePassForwardOutputPrimitive<B> = Wkv7StatePassForwardOutput<FloatTensor<B>>;
pub type Wkv7StatePassBackwardOutputPrimitive<B> = Wkv7StateBackwardOutput<FloatTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait Wkv7StatePassBackend: Backend {
    fn wkv7_statepass_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7StatePassForwardOutputPrimitive<Self>;

    fn wkv7_statepass_backward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        state: FloatTensor<Self>,
        removal_state: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        final_state_grad: FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7StatePassBackwardOutputPrimitive<Self>;
}

pub trait Wkv7StatePassAutodiffBackend: Wkv7StatePassBackend + AutodiffBackend {}

#[allow(clippy::too_many_arguments)]
pub fn wkv7_statepass_forward<B: Wkv7StatePassBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    chunk_len: usize,
) -> Wkv7StatePassForwardOutputTensor<B> {
    let output = B::wkv7_statepass_forward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        initial_state.into_primitive().tensor(),
        chunk_len,
    );

    Wkv7StatePassForwardOutput {
        state: Tensor::from_primitive(TensorPrimitive::Float(output.state)),
        removal_state: Tensor::from_primitive(TensorPrimitive::Float(output.removal_state)),
        output: Tensor::from_primitive(TensorPrimitive::Float(output.output)),
        final_state: Tensor::from_primitive(TensorPrimitive::Float(output.final_state)),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn wkv7_statepass_backward<B: Wkv7StatePassBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    state: Tensor<B, 4>,
    removal_state: Tensor<B, 4>,
    output_grad: Tensor<B, 4>,
    final_state_grad: Tensor<B, 4>,
    chunk_len: usize,
) -> Wkv7StatePassBackwardOutputTensor<B> {
    let output = B::wkv7_statepass_backward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        state.into_primitive().tensor(),
        removal_state.into_primitive().tensor(),
        output_grad.into_primitive().tensor(),
        final_state_grad.into_primitive().tensor(),
        chunk_len,
    );

    Wkv7StateBackwardOutput {
        weight_decay_grad: Tensor::from_primitive(TensorPrimitive::Float(output.weight_decay_grad)),
        receptance_grad: Tensor::from_primitive(TensorPrimitive::Float(output.receptance_grad)),
        key_grad: Tensor::from_primitive(TensorPrimitive::Float(output.key_grad)),
        value_grad: Tensor::from_primitive(TensorPrimitive::Float(output.value_grad)),
        removal_grad: Tensor::from_primitive(TensorPrimitive::Float(output.removal_grad)),
        replacement_grad: Tensor::from_primitive(TensorPrimitive::Float(output.replacement_grad)),
        initial_state_grad: Tensor::from_primitive(TensorPrimitive::Float(
            output.initial_state_grad,
        )),
    }
}
