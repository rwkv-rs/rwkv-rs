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
