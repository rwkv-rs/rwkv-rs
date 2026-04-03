use burn::{
    backend::{
        Autodiff,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
    },
    prelude::Int,
    tensor::{
        IndexingUpdateOp,
        Tensor,
        TensorMetadata,
        TensorPrimitive,
        ops::{FloatTensor, IntTensor},
    },
};

use crate::kernels::token_shift_diff::TokenShiftDiffBackend;

pub(crate) fn rwkv_token_shift_diff_autodiff<B: TokenShiftDiffBackend, C: CheckpointStrategy>(
    embedded_context: FloatTensor<Autodiff<B, C>>,
    embedded_token_shift: FloatTensor<Autodiff<B, C>>,
    batch_ids: IntTensor<Autodiff<B, C>>,
    context_mask: FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>> {
    #[derive(Debug)]
    struct TokenShiftDiffBackward;

    impl<B: TokenShiftDiffBackend> Backward<B, 2> for TokenShiftDiffBackward {
        type State = (FloatTensor<B>, IntTensor<B>, usize);

        fn backward(
            self,
            ops: Ops<Self::State, 2>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let [node_embedded_context, node_embedded_token_shift] = ops.parents;
            let grad = grads.consume::<B>(&ops.node);
            let (context_mask, batch_ids, full_batch_size) = ops.state;

            let grad = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(grad));
            let context_mask = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(context_mask));
            let batch_ids = Tensor::<B, 1, Int>::new(batch_ids);

            let [active_batch_size, context_length, embedded_dim] = grad.dims();
            let device = grad.device();

            let masked_grad = grad.clone() * context_mask.clone().unsqueeze_dim(2);

            let shifted_next_grad = if context_length > 1 {
                Tensor::cat(
                    vec![
                        masked_grad.clone().slice([
                            0..active_batch_size,
                            1..context_length,
                            0..embedded_dim,
                        ]),
                        Tensor::zeros([active_batch_size, 1, embedded_dim], &device),
                    ],
                    1,
                )
            } else {
                Tensor::zeros([active_batch_size, 1, embedded_dim], &device)
            };

            let embedded_context_grad =
                -masked_grad.clone() + context_mask.clone().unsqueeze_dim(2) * shifted_next_grad;

            let prev_valid_mask = if context_length > 1 {
                Tensor::cat(
                    vec![
                        Tensor::zeros([active_batch_size, 1], &device),
                        context_mask
                            .clone()
                            .slice([0..active_batch_size, 0..(context_length - 1)]),
                    ],
                    1,
                )
            } else {
                Tensor::zeros([active_batch_size, 1], &device)
            };
            let use_external_shift = context_mask.clone()
                * (Tensor::ones([active_batch_size, context_length], &device) - prev_valid_mask);

            let active_token_shift_grad: Tensor<B, 2> = (masked_grad
                * use_external_shift.unsqueeze_dim(2))
            .sum_dim(1)
            .squeeze_dim(1);

            let embedded_token_shift_grad =
                Tensor::<B, 2>::zeros([full_batch_size, embedded_dim], &device).select_assign(
                    0,
                    batch_ids,
                    active_token_shift_grad,
                    IndexingUpdateOp::Add,
                );

            if let Some(node) = node_embedded_context {
                grads.register::<B>(node.id, embedded_context_grad.into_primitive().tensor());
            }
            if let Some(node) = node_embedded_token_shift {
                grads.register::<B>(node.id, embedded_token_shift_grad.into_primitive().tensor());
            }
        }
    }

    let full_batch_size = embedded_token_shift.shape()[0];

    match TokenShiftDiffBackward
        .prepare::<C>([
            embedded_context.node.clone(),
            embedded_token_shift.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let output = B::token_shift_diff(
                embedded_context.primitive.clone(),
                embedded_token_shift.primitive.clone(),
                batch_ids.clone(),
                context_mask.primitive.clone(),
            );
            prep.finish(
                (context_mask.primitive, batch_ids, full_batch_size),
                output.token_shifted_diff,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(
            B::token_shift_diff(
                embedded_context.primitive,
                embedded_token_shift.primitive,
                batch_ids,
                context_mask.primitive,
            )
            .token_shifted_diff,
        ),
    }
}
