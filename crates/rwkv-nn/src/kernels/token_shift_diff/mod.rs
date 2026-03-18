mod backward;
mod forward;
mod kernel;

use burn::backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy};
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
};

#[derive(Clone, Debug)]
pub struct TokenShiftDiffOutput<B: Backend> {
    pub token_shifted_diff: Tensor<B, 3>,
    pub next_token_shift: Tensor<B, 2>,
}

pub trait TokenShiftDiffBackend: Backend {
    fn token_shift_diff(
        embedded_context: FloatTensor<Self>,
        embedded_token_shift: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(name = "rwkv.infer.model.token_shift_diff", skip_all)
)]
pub fn token_shift_diff<B: TokenShiftDiffBackend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Option<Tensor<B, 2>>,
    context_mask: Option<Tensor<B, 2>>,
) -> TokenShiftDiffOutput<B> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();
    let device = embedded_context.device();

    let next_token_shift = if context_length == 0 {
        Tensor::zeros([batch_size, embedded_dim], &device)
    } else {
        embedded_context
            .clone()
            .slice([
                0..batch_size,
                (context_length - 1)..context_length,
                0..embedded_dim,
            ])
            .squeeze_dim(1)
    };

    if context_length == 0 {
        return TokenShiftDiffOutput {
            token_shifted_diff: embedded_context,
            next_token_shift,
        };
    }

    let embedded_token_shift =
        embedded_token_shift.unwrap_or_else(|| Tensor::zeros([batch_size, embedded_dim], &device));
    let context_mask =
        context_mask.unwrap_or_else(|| Tensor::ones([batch_size, context_length], &device));

    let token_shifted_diff = B::token_shift_diff(
        embedded_context.into_primitive().tensor(),
        embedded_token_shift.into_primitive().tensor(),
        context_mask.into_primitive().tensor(),
    );

    TokenShiftDiffOutput {
        token_shifted_diff: Tensor::from_primitive(TensorPrimitive::Float(token_shifted_diff)),
        next_token_shift,
    }
}

impl<B: TokenShiftDiffBackend, C: CheckpointStrategy> TokenShiftDiffBackend for Autodiff<B, C> {
    fn token_shift_diff(
        embedded_context: FloatTensor<Self>,
        embedded_token_shift: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        backward::rwkv_token_shift_diff_autodiff::<B, C>(
            embedded_context,
            embedded_token_shift,
            context_mask,
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

    fn token_shift_diff_reference<B: Backend>(
        embedded_context: Tensor<B, 3>,
        embedded_token_shift: Tensor<B, 2>,
        context_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch_size, context_length, embedded_dim] = embedded_context.dims();
        if context_length == 0 {
            return embedded_context;
        }

        let shifted = if context_length == 1 {
            embedded_token_shift.clone().unsqueeze_dim(1)
        } else {
            Tensor::cat(
                vec![
                    embedded_token_shift.clone().unsqueeze_dim(1),
                    embedded_context.clone().slice([
                        0..batch_size,
                        0..(context_length - 1),
                        0..embedded_dim,
                    ]),
                ],
                1,
            )
        };

        let prev_valid_mask = if context_length == 1 {
            Tensor::zeros([batch_size, 1], &embedded_context.device())
        } else {
            Tensor::cat(
                vec![
                    Tensor::zeros([batch_size, 1], &embedded_context.device()),
                    context_mask
                        .clone()
                        .slice([0..batch_size, 0..(context_length - 1)]),
                ],
                1,
            )
        };

        let use_external_shift =
            Tensor::ones([batch_size, context_length], &embedded_context.device())
                - prev_valid_mask;
        let external_shift = embedded_token_shift.unsqueeze_dim(1);
        let prev =
            shifted.clone() + use_external_shift.unsqueeze_dim(2) * (external_shift - shifted);

        (prev - embedded_context) * context_mask.unsqueeze_dim(2)
    }

    #[test]
    fn token_shift_diff_matches_reference() {
        let device = Default::default();
        let embedded_context = Tensor::<TestBackend, 3>::from_floats(
            [
                [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 4.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            &device,
        );
        let embedded_token_shift =
            Tensor::<TestBackend, 2>::from_floats([[10.0, 20.0, 30.0], [1.0, 1.0, 1.0]], &device);
        let context_mask =
            Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], &device);

        let output = token_shift_diff(
            embedded_context.clone(),
            Some(embedded_token_shift.clone()),
            Some(context_mask.clone()),
        );
        let expected_diff = token_shift_diff_reference(
            embedded_context.clone(),
            embedded_token_shift,
            context_mask,
        );
        let expected_next = embedded_context.slice([0..2, 2..3, 0..3]).reshape([2, 3]);

        output
            .token_shifted_diff
            .into_data()
            .assert_approx_eq::<f32>(&expected_diff.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
        output
            .next_token_shift
            .into_data()
            .assert_approx_eq::<f32>(&expected_next.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }

    #[test]
    fn token_shift_diff_backward_matches_reference() {
        let device = Default::default();

        let embedded_context = Tensor::<TestAutodiffBackend, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            &device,
        )
        .require_grad();
        let embedded_token_shift =
            Tensor::<TestAutodiffBackend, 2>::from_floats([[10.0, 20.0]], &device).require_grad();
        let context_mask =
            Tensor::<TestAutodiffBackend, 2>::from_floats([[0.0, 1.0, 1.0]], &device);

        let custom = token_shift_diff(
            embedded_context.clone(),
            Some(embedded_token_shift.clone()),
            Some(context_mask.clone()),
        );
        let custom_grads = custom.token_shifted_diff.sum().backward();

        let ref_context = Tensor::<TestAutodiffBackend, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            &device,
        )
        .require_grad();
        let ref_shift =
            Tensor::<TestAutodiffBackend, 2>::from_floats([[10.0, 20.0]], &device).require_grad();
        let ref_mask = Tensor::<TestAutodiffBackend, 2>::from_floats([[0.0, 1.0, 1.0]], &device);
        let reference =
            token_shift_diff_reference(ref_context.clone(), ref_shift.clone(), ref_mask);
        let ref_grads = reference.sum().backward();

        embedded_context
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_context.grad(&ref_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        embedded_token_shift
            .grad(&custom_grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &ref_shift.grad(&ref_grads).unwrap().to_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
    }
}
