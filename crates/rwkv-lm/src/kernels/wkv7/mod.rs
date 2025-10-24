mod backward;
mod forward;
mod kernel;

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, backend::AutodiffBackend, ops::FloatTensor},
};

/// Interface matching the CUDA kernel described in model.py L46-L68.

/// WKV7 Backend trait that extends the Burn backend trait

pub trait Wkv7Backend: Backend {
    fn wkv7_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: Option<FloatTensor<Self>>,
        chunk_len: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    /// WKV7 backward operation

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
    );
}

/// Autodiff backend trait that combines WKV7 operations with autodiff
/// capabilities

pub trait Wkv7AutodiffBackend: Wkv7Backend + AutodiffBackend {}

/// High-level WKV7 forward function

pub fn wkv7_forward<B: Wkv7Backend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    initial_state: Option<Tensor<B, 4>>,
    chunk_len: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let initial_state_primitive = initial_state.map(|s| s.into_primitive().tensor());

    let (state, removal_state, output) = B::wkv7_forward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        initial_state_primitive,
        chunk_len,
    );

    (
        Tensor::from_primitive(TensorPrimitive::Float(state)),
        Tensor::from_primitive(TensorPrimitive::Float(removal_state)),
        Tensor::from_primitive(TensorPrimitive::Float(output)),
    )
}
