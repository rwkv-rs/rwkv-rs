mod backward;
mod forward;
mod kernel;

use burn::backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy};
use burn::{
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

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, Cpu};
    use burn::tensor::Tolerance;

    use super::*;

    type TestBackend = Cpu<f32, i32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;

    fn reference_addcmul5<B: Backend>(
        base: Tensor<B, 3>,
        diff: Tensor<B, 3>,
        receptance_scale: Tensor<B, 3>,
        weight_decay_scale: Tensor<B, 3>,
        key_scale: Tensor<B, 3>,
        value_scale: Tensor<B, 3>,
        learning_rate_scale: Tensor<B, 3>,
    ) -> Addcmul5Output<Tensor<B, 3>> {
        Addcmul5Output {
            receptance_input: base.clone() + diff.clone() * receptance_scale,
            weight_decay_input: base.clone() + diff.clone() * weight_decay_scale,
            key_input: base.clone() + diff.clone() * key_scale,
            value_input: base.clone() + diff.clone() * value_scale,
            learning_rate_input: base + diff * learning_rate_scale,
        }
    }

    fn assert_addcmul5_output_close<B: Backend>(
        output: Addcmul5Output<Tensor<B, 3>>,
        reference: Addcmul5Output<Tensor<B, 3>>,
    ) {
        output.receptance_input.into_data().assert_approx_eq::<f32>(
            &reference.receptance_input.into_data(),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
        output
            .weight_decay_input
            .into_data()
            .assert_approx_eq::<f32>(
                &reference.weight_decay_input.into_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        output.key_input.into_data().assert_approx_eq::<f32>(
            &reference.key_input.into_data(),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
        output.value_input.into_data().assert_approx_eq::<f32>(
            &reference.value_input.into_data(),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
        output
            .learning_rate_input
            .into_data()
            .assert_approx_eq::<f32>(
                &reference.learning_rate_input.into_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
    }

    #[test]
    fn addcmul_matches_reference() {
        let device = Default::default();
        let base = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device);
        let diff = Tensor::<TestBackend, 3>::from_floats([[[0.5, 1.0], [1.5, 2.0]]], &device);
        let scale = Tensor::<TestBackend, 3>::from_floats([[[2.0, 3.0]]], &device);

        let output = addcmul(base.clone(), diff.clone(), scale.clone());
        let reference = base + diff * scale;

        output
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }

    #[test]
    fn addcmul5_matches_reference() {
        let device = Default::default();

        let base = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device);
        let diff = Tensor::<TestBackend, 3>::from_floats([[[0.5, 1.0], [1.5, 2.0]]], &device);
        let receptance_scale = Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0]]], &device);
        let weight_decay_scale = Tensor::<TestBackend, 3>::from_floats([[[0.5, 1.5]]], &device);
        let key_scale = Tensor::<TestBackend, 3>::from_floats([[[2.0, 0.5]]], &device);
        let value_scale = Tensor::<TestBackend, 3>::from_floats([[[1.5, 1.0]]], &device);
        let learning_rate_scale = Tensor::<TestBackend, 3>::from_floats([[[0.25, 0.75]]], &device);

        let output = addcmul5(
            base.clone(),
            diff.clone(),
            receptance_scale.clone(),
            weight_decay_scale.clone(),
            key_scale.clone(),
            value_scale.clone(),
            learning_rate_scale.clone(),
        );
        let reference = reference_addcmul5(
            base,
            diff,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
        );

        assert_addcmul5_output_close(output, reference);
    }

    #[test]
    fn addcmul5_backward_matches_reference() {
        let device = Default::default();

        let base =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device)
                .require_grad();
        let diff =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[0.5, 1.0], [1.5, 2.0]]], &device)
                .require_grad();
        let receptance_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[1.0, 2.0]]], &device).require_grad();
        let weight_decay_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[0.5, 1.5]]], &device).require_grad();
        let key_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[2.0, 0.5]]], &device).require_grad();
        let value_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[1.5, 1.0]]], &device).require_grad();
        let learning_rate_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[0.25, 0.75]]], &device).require_grad();

        let custom = addcmul5(
            base.clone(),
            diff.clone(),
            receptance_scale.clone(),
            weight_decay_scale.clone(),
            key_scale.clone(),
            value_scale.clone(),
            learning_rate_scale.clone(),
        );
        let custom_loss = custom.receptance_input.sum()
            + custom.weight_decay_input.sum() * 2.0
            + custom.key_input.sum() * 3.0
            + custom.value_input.sum() * 4.0
            + custom.learning_rate_input.sum() * 5.0;
        let custom_grads = custom_loss.backward();

        let ref_base =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device)
                .require_grad();
        let ref_diff =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[0.5, 1.0], [1.5, 2.0]]], &device)
                .require_grad();
        let ref_receptance_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[1.0, 2.0]]], &device).require_grad();
        let ref_weight_decay_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[0.5, 1.5]]], &device).require_grad();
        let ref_key_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[2.0, 0.5]]], &device).require_grad();
        let ref_value_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[1.5, 1.0]]], &device).require_grad();
        let ref_learning_rate_scale =
            Tensor::<TestAutodiffBackend, 3>::from_floats([[[0.25, 0.75]]], &device).require_grad();

        let reference = reference_addcmul5(
            ref_base.clone(),
            ref_diff.clone(),
            ref_receptance_scale.clone(),
            ref_weight_decay_scale.clone(),
            ref_key_scale.clone(),
            ref_value_scale.clone(),
            ref_learning_rate_scale.clone(),
        );
        let reference_loss = reference.receptance_input.sum()
            + reference.weight_decay_input.sum() * 2.0
            + reference.key_input.sum() * 3.0
            + reference.value_input.sum() * 4.0
            + reference.learning_rate_input.sum() * 5.0;
        let reference_grads = reference_loss.backward();

        base.grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_base.grad(&reference_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        diff.grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_diff.grad(&reference_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        receptance_scale
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_receptance_scale
                    .grad(&reference_grads)
                    .unwrap()
                    .to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        weight_decay_scale
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_weight_decay_scale
                    .grad(&reference_grads)
                    .unwrap()
                    .to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        key_scale
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_key_scale.grad(&reference_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        value_scale
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_value_scale.grad(&reference_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        learning_rate_scale
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_learning_rate_scale
                    .grad(&reference_grads)
                    .unwrap()
                    .to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
    }
}
