mod forward;
mod host;
mod kernel;

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
};

/// Forward output for inference-only WKV7 kernel.
///
/// - `output`: [batch_size, sequence_length, num_heads, head_size]
/// - `final_state`: [batch_size, num_heads, head_size, head_size]
#[derive(Clone, Debug)]
pub struct Wkv7InferenceForwardOutput<T> {
    pub output: T,
    pub final_state: T,
}

pub type Wkv7InferenceForwardOutputTensor<B> = Wkv7InferenceForwardOutput<Tensor<B, 4>>;
pub type Wkv7InferenceForwardOutputPrimitive<B> = Wkv7InferenceForwardOutput<FloatTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait Wkv7InferenceBackend: Backend {
    fn wkv7_inference_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
    ) -> Wkv7InferenceForwardOutputPrimitive<Self>;
}

#[allow(clippy::too_many_arguments)]
pub fn wkv7_inference_forward<B: Wkv7InferenceBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
) -> Wkv7InferenceForwardOutputTensor<B> {
    let output = B::wkv7_inference_forward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        initial_state.into_primitive().tensor(),
    );

    Wkv7InferenceForwardOutput {
        output: Tensor::from_primitive(TensorPrimitive::Float(output.output)),
        final_state: Tensor::from_primitive(TensorPrimitive::Float(output.final_state)),
    }
}

