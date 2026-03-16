use burn::{
    backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::{
        Tensor, TensorMetadata, TensorPrimitive,
        backend::{AutodiffBackend, Backend},
    },
};
#[cfg(feature = "fusion")]
use burn_fusion::{Fusion, FusionBackend};

use crate::kernels::backend::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

pub trait L2WrapBackend: Backend {
    fn apply_l2wrap(
        logits: <Self as Backend>::FloatTensorPrimitive,
        num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive;
}

impl<B: Backend, C: CheckpointStrategy> L2WrapBackend for Autodiff<B, C> {
    fn apply_l2wrap(
        logits: <Self as Backend>::FloatTensorPrimitive,
        num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive {
        #[derive(Debug)]
        struct L2WrapBackward;

        impl<B: Backend> Backward<B, 1> for L2WrapBackward {
            type State = (B::FloatTensorPrimitive, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_logits] = ops.parents;

                // Gradient flowing from the loss into the wrapped logits.
                let grad_output = grads.consume::<B>(&ops.node);
                let (saved_logits, num_tokens_per_batch) = ops.state;

                if let Some(node) = node_logits {
                    if num_tokens_per_batch == 0 {
                        grads.register::<B>(node.id, grad_output);
                        return;
                    }

                    let shape = saved_logits.shape();
                    let last_dim = shape.num_dims() - 1;

                    // Inject factor * max(logits) into the gradient at the argmax position.
                    let (max_vals, max_ids) = B::float_max_dim_with_indices(saved_logits, last_dim);
                    let inject_vals =
                        B::float_mul_scalar(max_vals, (1e-4 / num_tokens_per_batch as f64).into());
                    let grad_adjusted =
                        B::float_scatter_add(last_dim, grad_output, max_ids, inject_vals);

                    grads.register::<B>(node.id, grad_adjusted);
                }
            }
        }

        match L2WrapBackward
            .prepare::<C>([logits.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let logits_state = logits.primitive.clone();
                let output = logits.primitive.clone();
                prep.finish((logits_state, num_tokens_per_batch), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(logits.primitive),
        }
    }
}

impl<R, F, I, BT> L2WrapBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn apply_l2wrap(
        logits: <Self as Backend>::FloatTensorPrimitive,
        _num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive {
        logits
    }
}

#[cfg(feature = "fusion")]
impl<B: FusionBackend> L2WrapBackend for Fusion<B> {
    fn apply_l2wrap(
        logits: <Self as Backend>::FloatTensorPrimitive,
        _num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive {
        logits
    }
}

pub fn l2wrap<B: L2WrapBackend>(logits: Tensor<B, 2>, num_tokens_per_batch: usize) -> Tensor<B, 2> {
    let output = B::apply_l2wrap(logits.into_primitive().tensor(), num_tokens_per_batch);
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub trait L2WrapAutodiffBackend: L2WrapBackend + AutodiffBackend {}

impl<B: L2WrapBackend + AutodiffBackend> L2WrapAutodiffBackend for B {}
