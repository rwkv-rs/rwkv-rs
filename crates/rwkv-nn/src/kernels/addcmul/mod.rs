mod backward;
mod forward;
mod kernel;

use burn::{
    backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy},
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
};

#[derive(Clone, Debug)]
pub struct Addcmul5Output<T> {
    pub receptance_input: T,
    pub weight_decay_input: T,
    pub key_input: T,
    pub value_input: T,
    pub learning_rate_input: T,
}

pub type Addcmul5OutputTensor<B> = Addcmul5Output<Tensor<B, 3>>;
pub type Addcmul5OutputPrimitive<B> = Addcmul5Output<FloatTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait AddcmulBackend: Backend {
    fn addcmul(
        base: FloatTensor<Self>,
        diff: FloatTensor<Self>,
        scale: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    fn addcmul5(
        base: FloatTensor<Self>,
        diff: FloatTensor<Self>,
        receptance_scale: FloatTensor<Self>,
        weight_decay_scale: FloatTensor<Self>,
        key_scale: FloatTensor<Self>,
        value_scale: FloatTensor<Self>,
        learning_rate_scale: FloatTensor<Self>,
    ) -> Addcmul5OutputPrimitive<Self>;
}

pub fn addcmul<B: AddcmulBackend>(
    base: Tensor<B, 3>,
    diff: Tensor<B, 3>,
    scale: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::addcmul(
        base.into_primitive().tensor(),
        diff.into_primitive().tensor(),
        scale.into_primitive().tensor(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

#[allow(clippy::too_many_arguments)]
pub fn addcmul5<B: AddcmulBackend>(
    base: Tensor<B, 3>,
    diff: Tensor<B, 3>,
    receptance_scale: Tensor<B, 3>,
    weight_decay_scale: Tensor<B, 3>,
    key_scale: Tensor<B, 3>,
    value_scale: Tensor<B, 3>,
    learning_rate_scale: Tensor<B, 3>,
) -> Addcmul5OutputTensor<B> {
    let output = B::addcmul5(
        base.into_primitive().tensor(),
        diff.into_primitive().tensor(),
        receptance_scale.into_primitive().tensor(),
        weight_decay_scale.into_primitive().tensor(),
        key_scale.into_primitive().tensor(),
        value_scale.into_primitive().tensor(),
        learning_rate_scale.into_primitive().tensor(),
    );

    Addcmul5Output {
        receptance_input: Tensor::from_primitive(TensorPrimitive::Float(output.receptance_input)),
        weight_decay_input: Tensor::from_primitive(TensorPrimitive::Float(
            output.weight_decay_input,
        )),
        key_input: Tensor::from_primitive(TensorPrimitive::Float(output.key_input)),
        value_input: Tensor::from_primitive(TensorPrimitive::Float(output.value_input)),
        learning_rate_input: Tensor::from_primitive(TensorPrimitive::Float(
            output.learning_rate_input,
        )),
    }
}

impl<B: AddcmulBackend, C: CheckpointStrategy> AddcmulBackend for Autodiff<B, C> {
    fn addcmul(
        base: FloatTensor<Self>,
        diff: FloatTensor<Self>,
        scale: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        backward::addcmul_autodiff::<B, C>(base, diff, scale)
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
        backward::addcmul5_autodiff::<B, C>(
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
