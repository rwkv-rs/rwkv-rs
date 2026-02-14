mod forward;
mod host;
mod kernel;


use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
};
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::tensor::backend::AutodiffBackend;

/// Forward output for inference-only WKV7 kernel.
///
/// - `output`: [batch_size, context_length, num_heads, head_size]
/// - `final_state`: [batch_size, num_heads, head_size, head_size]
#[derive(Clone, Debug)]
pub struct Wkv7InferForwardOutput<T> {
    pub output: T,
    pub final_state: T,
}

pub type Wkv7InferForwardOutputTensor<B> = Wkv7InferForwardOutput<Tensor<B, 4>>;
pub type Wkv7InferForwardOutputPrimitive<B> = Wkv7InferForwardOutput<FloatTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait Wkv7InferBackend: Backend {
    /// Inference forward.
    ///
    /// - `context_mask`: [batch_size, context_length] with values 0/1.
    ///   When 0, the corresponding timestep is treated as padding and must be a strict no-op
    ///   for the internal state.
    fn wkv7_infer_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> Wkv7InferForwardOutputPrimitive<Self>;
}

#[allow(clippy::too_many_arguments)]
pub fn wkv7_infer_forward<B: Wkv7InferBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    initial_state: Tensor<B, 4>,
    context_mask: Tensor<B, 2>,
) -> Wkv7InferForwardOutputTensor<B> {
    let output = B::wkv7_infer_forward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        initial_state.into_primitive().tensor(),
        context_mask.into_primitive().tensor(),
    );

    Wkv7InferForwardOutput {
        output: Tensor::from_primitive(TensorPrimitive::Float(output.output)),
        final_state: Tensor::from_primitive(TensorPrimitive::Float(output.final_state)),
    }
}


impl<B: Wkv7InferBackend, C: CheckpointStrategy> Wkv7InferBackend for Autodiff<B, C> {
    fn wkv7_infer_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> Wkv7InferForwardOutput<FloatTensor<Self>> {
        let output = B::wkv7_infer_forward(
            weight_decay.primitive,
            receptance.primitive,
            key.primitive,
            value.primitive,
            removal.primitive,
            replacement.primitive,
            initial_state.primitive,
            context_mask.primitive,
        );

        Wkv7InferForwardOutput {
            output: Autodiff::<B, C>::from_inner(output.output),
            final_state: Autodiff::<B, C>::from_inner(output.final_state),
        }
    }
}
