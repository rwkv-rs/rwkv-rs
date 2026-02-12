use burn::{
    backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::{
        Element, ElementConversion, FloatDType,
        FloatDType::F32,
        Tensor, TensorMetadata, TensorPrimitive,
        backend::{AutodiffBackend, Backend},
    },
};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
#[cfg(feature = "fusion")]
use burn_fusion::{Fusion, FusionBackend};
pub trait L2WrapBackend: Backend {
    fn apply_l2wrap(
        loss: <Self as Backend>::FloatTensorPrimitive,
        logits: <Self as Backend>::FloatTensorPrimitive,
        num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive;
}

impl<B: Backend, C: CheckpointStrategy> L2WrapBackend for Autodiff<B, C> {
    fn apply_l2wrap(
        loss: <Self as Backend>::FloatTensorPrimitive,
        logits: <Self as Backend>::FloatTensorPrimitive,
        num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive {
        #[derive(Debug)]

        struct L2WrapBackward;

        impl<B: Backend> Backward<B, 2> for L2WrapBackward {
            type State = (B::FloatTensorPrimitive, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_loss, node_logits] = ops.parents;

                let grad_output = grads.consume::<B>(&ops.node);

                let (saved_logits, num_tokens_per_batch) = ops.state;

                if let Some(node) = node_loss {
                    grads.register::<B>(node.id, grad_output.clone());
                }

                if let Some(node) = node_logits {
                    if num_tokens_per_batch == 0 {
                        return;
                    }
                    let factor = B::FloatElem::from_elem(1e-4 / num_tokens_per_batch as f64);

                    // Upcast to f32 for numerically stable max/scatter, then downcast back
                    // Determine original dtype and device
                    let shape = saved_logits.shape();

                    let last_dim = shape.dims.len() - 1;

                    // logits -> f32
                    let logits_f32 = B::float_cast(saved_logits.clone(), F32);

                    let (max_vals_f32, max_ids) =
                        B::float_max_dim_with_indices(logits_f32.clone(), last_dim);

                    let zeros_f32 =
                        B::float_zeros(shape.clone(), &B::float_device(&max_vals_f32), F32);

                    let inject_vals_f32 = B::float_mul_scalar(max_vals_f32, factor);

                    let logits_grad_f32 =
                        B::float_scatter_add(last_dim, zeros_f32, max_ids, inject_vals_f32);

                    // Cast gradient back to original dtype of backend float
                    let logits_grad =
                        B::float_cast(logits_grad_f32, FloatDType::from(B::FloatElem::dtype()));

                    grads.register::<B>(node.id, logits_grad);
                }
            }
        }

        // Standard Burn autodiff registration flow
        match L2WrapBackward
            .prepare::<C>([loss.node.clone(), logits.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let logits_state = logits.primitive.clone();

                let output = loss.primitive.clone();

                prep.finish((logits_state, num_tokens_per_batch), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(loss.primitive),
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
        loss: <Self as Backend>::FloatTensorPrimitive,
        _logits: <Self as Backend>::FloatTensorPrimitive,
        _num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive {
        loss
    }
}

#[cfg(feature = "fusion")]
impl<B: FusionBackend> L2WrapBackend for Fusion<B> {
    fn apply_l2wrap(
        loss: <Self as Backend>::FloatTensorPrimitive,
        _logits: <Self as Backend>::FloatTensorPrimitive,
        _num_tokens_per_batch: usize,
    ) -> <Self as Backend>::FloatTensorPrimitive {
        loss
    }
}

/// High-level API for L2Wrap application
pub fn l2wrap<B: L2WrapBackend>(
    loss: Tensor<B, 1>,
    logits: Tensor<B, 2>,
    num_tokens_per_batch: usize,
) -> Tensor<B, 1> {
    let output = B::apply_l2wrap(
        loss.into_primitive().tensor(),
        logits.into_primitive().tensor(),
        num_tokens_per_batch,
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Marker trait for backends that support both L2Wrap and AutodiffBackend
pub trait L2WrapAutodiffBackend: L2WrapBackend + AutodiffBackend {}

/// Blanket implementation
impl<B: L2WrapBackend + AutodiffBackend> L2WrapAutodiffBackend for B {}
