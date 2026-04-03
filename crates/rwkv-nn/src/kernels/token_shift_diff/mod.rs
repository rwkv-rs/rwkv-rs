mod backward;
mod forward;
mod kernel;

use burn::{
    backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy},
    prelude::{Backend, Int},
    tensor::{
        IndexingUpdateOp,
        Tensor,
        TensorPrimitive,
        ops::{FloatTensor, IntTensor},
    },
};

#[derive(Clone, Debug)]
pub struct TokenShiftDiffOutput<B: Backend> {
    pub token_shifted_diff: Tensor<B, 3>,
    pub next_token_shift: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct TokenShiftDiffPrimitiveOutput<B: Backend> {
    pub token_shifted_diff: FloatTensor<B>,
    pub next_token_shift: FloatTensor<B>,
}

pub trait TokenShiftDiffBackend: Backend {
    fn token_shift_diff(
        embedded_context: FloatTensor<Self>,
        embedded_token_shift: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> TokenShiftDiffPrimitiveOutput<Self>;
}

#[cfg_attr(not(test), allow(dead_code))]
fn token_shift_diff_reference<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Tensor<B, 2>,
    batch_ids: Tensor<B, 1, Int>,
    context_mask: Tensor<B, 2>,
) -> TokenShiftDiffOutput<B> {
    let [active_batch_size, context_length, embedded_dim] = embedded_context.dims();
    let [_full_batch_size, shift_embedded_dim] = embedded_token_shift.dims();
    let device = embedded_context.device();

    debug_assert_eq!(
        embedded_dim, shift_embedded_dim,
        "embedded_token_shift feature dim mismatch"
    );

    if context_length == 0 {
        return TokenShiftDiffOutput {
            token_shifted_diff: embedded_context,
            next_token_shift: embedded_token_shift,
        };
    }

    let active_token_shift = embedded_token_shift.clone().select(0, batch_ids.clone());

    let shifted = if context_length == 1 {
        active_token_shift.clone().unsqueeze_dim(1)
    } else {
        Tensor::cat(
            vec![
                active_token_shift.clone().unsqueeze_dim(1),
                embedded_context.clone().slice([
                    0..active_batch_size,
                    0..(context_length - 1),
                    0..embedded_dim,
                ]),
            ],
            1,
        )
    };

    let prev_valid_mask = if context_length == 1 {
        Tensor::zeros([active_batch_size, 1], &device)
    } else {
        Tensor::cat(
            vec![
                Tensor::zeros([active_batch_size, 1], &device),
                context_mask
                    .clone()
                    .slice([0..active_batch_size, 0..(context_length - 1)]),
            ],
            1,
        )
    };

    let use_external_shift =
        Tensor::ones([active_batch_size, context_length], &device) - prev_valid_mask;
    let prev = shifted.clone()
        + use_external_shift.unsqueeze_dim(2)
            * (active_token_shift.clone().unsqueeze_dim(1) - shifted);
    let token_shifted_diff =
        (prev - embedded_context.clone()) * context_mask.clone().unsqueeze_dim(2);

    TokenShiftDiffOutput {
        token_shifted_diff,
        next_token_shift: next_token_shift_reference(
            embedded_context,
            embedded_token_shift,
            batch_ids,
            context_mask,
        ),
    }
}

fn next_token_shift_reference<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Tensor<B, 2>,
    batch_ids: Tensor<B, 1, Int>,
    context_mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [active_batch_size, context_length, embedded_dim] = embedded_context.dims();
    let [full_batch_size, shift_embedded_dim] = embedded_token_shift.dims();
    let device = embedded_context.device();

    debug_assert_eq!(
        embedded_dim, shift_embedded_dim,
        "embedded_token_shift feature dim mismatch"
    );

    if context_length == 0 {
        return embedded_token_shift;
    }

    let active_token_shift = embedded_token_shift.clone().select(0, batch_ids.clone());
    let last_token = embedded_context
        .slice([
            0..active_batch_size,
            (context_length - 1)..context_length,
            0..embedded_dim,
        ])
        .squeeze_dim(1);
    let last_mask: Tensor<B, 1> = context_mask
        .clone()
        .slice([0..active_batch_size, (context_length - 1)..context_length])
        .squeeze_dim(1);
    let next_active_token_shift =
        active_token_shift.clone() + last_mask.unsqueeze_dim(1) * (last_token - active_token_shift);

    let selected_rows = Tensor::<B, 1>::zeros([full_batch_size], &device).select_assign(
        0,
        batch_ids.clone(),
        Tensor::ones([active_batch_size], &device),
        IndexingUpdateOp::Add,
    );
    (embedded_token_shift.clone()
        * (Tensor::ones([full_batch_size], &device) - selected_rows).unsqueeze_dim(1))
    .select_assign(0, batch_ids, next_active_token_shift, IndexingUpdateOp::Add)
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(name = "rwkv.infer.model.token_shift_diff", skip_all)
)]
pub fn token_shift_diff<B: TokenShiftDiffBackend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Option<Tensor<B, 2>>,
    batch_ids: Tensor<B, 1, Int>,
    context_mask: Option<Tensor<B, 2>>,
) -> TokenShiftDiffOutput<B> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();
    let device = embedded_context.device();

    debug_assert_eq!(
        batch_ids.dims(),
        [batch_size],
        "batch_ids shape mismatch with embedded_context batch size"
    );

    let embedded_token_shift =
        embedded_token_shift.unwrap_or_else(|| Tensor::zeros([batch_size, embedded_dim], &device));
    let context_mask =
        context_mask.unwrap_or_else(|| Tensor::ones([batch_size, context_length], &device));

    let output = if context_length == 0 {
        TokenShiftDiffPrimitiveOutput {
            token_shifted_diff: embedded_context.into_primitive().tensor(),
            next_token_shift: embedded_token_shift.into_primitive().tensor(),
        }
    } else {
        B::token_shift_diff(
            embedded_context.into_primitive().tensor(),
            embedded_token_shift.into_primitive().tensor(),
            batch_ids.into_primitive(),
            context_mask.into_primitive().tensor(),
        )
    };

    TokenShiftDiffOutput {
        token_shifted_diff: Tensor::from_primitive(TensorPrimitive::Float(
            output.token_shifted_diff,
        )),
        next_token_shift: Tensor::from_primitive(TensorPrimitive::Float(output.next_token_shift)),
    }
}

impl<B: TokenShiftDiffBackend, C: CheckpointStrategy> TokenShiftDiffBackend for Autodiff<B, C> {
    fn token_shift_diff(
        embedded_context: FloatTensor<Self>,
        embedded_token_shift: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> TokenShiftDiffPrimitiveOutput<Self> {
        let embedded_context_tensor =
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(embedded_context.clone()));
        let embedded_token_shift_tensor =
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(embedded_token_shift.clone()));
        let batch_ids_tensor = Tensor::<Self, 1, Int>::new(batch_ids.clone());
        let context_mask_tensor =
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(context_mask.clone()));
        let context_length = embedded_context_tensor.dims()[1];

        // The fast backend path may update the full token-shift buffer in place to avoid an
        // extra full-buffer copy on inference. Compute the functional `next_token_shift` result
        // first so autodiff still returns the mathematically correct second output.
        let next_token_shift = next_token_shift_reference(
            embedded_context_tensor.clone(),
            embedded_token_shift_tensor,
            batch_ids_tensor,
            context_mask_tensor,
        );
        let token_shifted_diff = if context_length == 0 {
            embedded_context_tensor.clone()
        } else {
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(
                backward::rwkv_token_shift_diff_autodiff::<B, C>(
                    embedded_context,
                    embedded_token_shift,
                    batch_ids,
                    context_mask,
                ),
            ))
        };

        TokenShiftDiffPrimitiveOutput {
            token_shifted_diff: token_shifted_diff.into_primitive().tensor(),
            next_token_shift: next_token_shift.into_primitive().tensor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, Cpu},
        tensor::Tolerance,
    };

    use super::*;

    type TestBackend = Autodiff<Cpu<f32, i32>>;

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
        let embedded_token_shift = Tensor::<TestBackend, 2>::from_floats(
            [[10.0, 20.0, 30.0], [1.0, 1.0, 1.0], [100.0, 100.0, 100.0]],
            &device,
        );
        let batch_ids = Tensor::<TestBackend, 1, Int>::from_ints([1, 0], &device);
        let context_mask =
            Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], &device);

        // The fast path is allowed to recycle the token-shift buffer storage. Keep the reference
        // inputs independent so this test checks math rather than aliasing side effects.
        let expected = token_shift_diff_reference(
            embedded_context.clone(),
            embedded_token_shift.clone(),
            batch_ids.clone(),
            context_mask.clone(),
        );
        let output = token_shift_diff(
            embedded_context,
            Some(embedded_token_shift),
            batch_ids,
            Some(context_mask),
        );

        output
            .token_shifted_diff
            .into_data()
            .assert_approx_eq::<f32>(
                &expected.token_shifted_diff.into_data(),
                Tolerance::rel_abs(1e-5, 1e-5),
            );
        output.next_token_shift.into_data().assert_approx_eq::<f32>(
            &expected.next_token_shift.into_data(),
            Tolerance::rel_abs(1e-5, 1e-5),
        );
    }

    #[test]
    fn token_shift_diff_backward_matches_reference() {
        let device = Default::default();

        let embedded_context =
            Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], &device)
                .require_grad();
        let embedded_token_shift =
            Tensor::<TestBackend, 2>::from_floats([[10.0, 20.0], [30.0, 40.0]], &device)
                .require_grad();
        let batch_ids = Tensor::<TestBackend, 1, Int>::from_ints([1], &device);
        let context_mask = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 1.0]], &device);

        let custom = token_shift_diff(
            embedded_context.clone(),
            Some(embedded_token_shift.clone()),
            batch_ids.clone(),
            Some(context_mask.clone()),
        );
        let custom_grads = custom.token_shifted_diff.sum().backward();

        let ref_context =
            Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], &device)
                .require_grad();
        let ref_shift =
            Tensor::<TestBackend, 2>::from_floats([[10.0, 20.0], [30.0, 40.0]], &device)
                .require_grad();
        let ref_batch_ids = Tensor::<TestBackend, 1, Int>::from_ints([1], &device);
        let ref_mask = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 1.0]], &device);
        let reference = token_shift_diff_reference(
            ref_context.clone(),
            ref_shift.clone(),
            ref_batch_ids,
            ref_mask,
        );
        let ref_grads = reference.token_shifted_diff.sum().backward();

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
