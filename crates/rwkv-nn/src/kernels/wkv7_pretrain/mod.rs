mod backward;
mod forward;

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, backend::AutodiffBackend, ops::FloatTensor},
};

use crate::kernels::wkv7_common::Wkv7ForwardOutput;
use crate::kernels::wkv7_common::Wkv7BackwardOutput;

pub type Wkv7PretrainForwardOutput<B> = Wkv7ForwardOutput<Tensor<B, 4>>;
pub type Wkv7PretrainBackwardOutput<B> = Wkv7BackwardOutput<Tensor<B, 4>>;

pub type Wkv7PretrainForwardOutputPrimitive<B> = Wkv7ForwardOutput<FloatTensor<B>>;
pub type Wkv7PretrainBackwardOutputPrimitive<B> = Wkv7BackwardOutput<FloatTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait Wkv7PretrainBackend: Backend {
    fn wkv7_pretrain_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7PretrainForwardOutputPrimitive<Self>;

    fn wkv7_pretrain_backward(
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
    ) -> Wkv7PretrainBackwardOutputPrimitive<Self>;
}

pub trait Wkv7PretrainAutodiffBackend: Wkv7PretrainBackend + AutodiffBackend {}

#[allow(clippy::too_many_arguments)]
pub fn wkv7_pretrain_forward<B: Wkv7PretrainBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    chunk_len: usize,
) -> Wkv7PretrainForwardOutput<B> {
    let output = B::wkv7_pretrain_forward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        chunk_len,
    );

    Wkv7ForwardOutput {
        state: Tensor::from_primitive(TensorPrimitive::Float(output.state)),
        removal_state: Tensor::from_primitive(TensorPrimitive::Float(output.removal_state)),
        output: Tensor::from_primitive(TensorPrimitive::Float(output.output)),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn wkv7_pretrain_backward<B: Wkv7PretrainBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    state: Tensor<B, 4>,
    removal_state: Tensor<B, 4>,
    output_grad: Tensor<B, 4>,
    chunk_len: usize,
) -> Wkv7PretrainBackwardOutput<B> {
    let output = B::wkv7_pretrain_backward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        state.into_primitive().tensor(),
        removal_state.into_primitive().tensor(),
        output_grad.into_primitive().tensor(),
        chunk_len,
    );

    Wkv7BackwardOutput {
        weight_decay_grad: Tensor::from_primitive(TensorPrimitive::Float(output.weight_decay_grad)),
        receptance_grad: Tensor::from_primitive(TensorPrimitive::Float(output.receptance_grad)),
        key_grad: Tensor::from_primitive(TensorPrimitive::Float(output.key_grad)),
        value_grad: Tensor::from_primitive(TensorPrimitive::Float(output.value_grad)),
        removal_grad: Tensor::from_primitive(TensorPrimitive::Float(output.removal_grad)),
        replacement_grad: Tensor::from_primitive(TensorPrimitive::Float(output.replacement_grad)),
    }
}
