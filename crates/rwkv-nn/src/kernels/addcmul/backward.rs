use burn::{
    backend::{
        Autodiff,
        autodiff::{
            NodeId,
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
    },
    tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
};

use crate::kernels::addcmul::{Addcmul5Output, Addcmul5OutputPrimitive, AddcmulBackend};

fn attach_addcmul_output<B: AddcmulBackend, C: CheckpointStrategy>(
    base: FloatTensor<Autodiff<B, C>>,
    diff: FloatTensor<Autodiff<B, C>>,
    scale: FloatTensor<Autodiff<B, C>>,
    output: FloatTensor<B>,
) -> FloatTensor<Autodiff<B, C>> {
    #[derive(Debug)]
    struct AddcmulBackward;

    impl<B: AddcmulBackend> Backward<B, 3> for AddcmulBackward {
        type State = (NodeId, FloatTensor<B>);

        fn backward(
            self,
            ops: Ops<Self::State, 3>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let [node_base, node_diff, node_scale] = ops.parents;
            let grad = grads.consume::<B>(&ops.node);
            let (diff_state, scale_saved) = ops.state;

            let diff = checkpointer.retrieve_node_output(diff_state);

            let grad = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(grad));
            let diff = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(diff));
            let scale = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(scale_saved));
            let embedded_dim = grad.dims()[2];

            let base_grad = grad.clone();
            let diff_grad = grad.clone() * scale.clone();
            let scale_grad = (grad * diff)
                .sum_dim(0)
                .sum_dim(1)
                .reshape([1, 1, embedded_dim]);

            if let Some(node) = node_base {
                grads.register::<B>(node.id, base_grad.into_primitive().tensor());
            }
            if let Some(node) = node_diff {
                grads.register::<B>(node.id, diff_grad.into_primitive().tensor());
            }
            if let Some(node) = node_scale {
                grads.register::<B>(node.id, scale_grad.into_primitive().tensor());
            }
        }
    }

    match AddcmulBackward
        .prepare::<C>([base.node.clone(), diff.node.clone(), scale.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            let diff_state = prep.checkpoint(&diff);
            prep.finish((diff_state, scale.primitive), output)
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

pub(crate) fn addcmul_autodiff<B: AddcmulBackend, C: CheckpointStrategy>(
    base: FloatTensor<Autodiff<B, C>>,
    diff: FloatTensor<Autodiff<B, C>>,
    scale: FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>> {
    let output = B::addcmul(
        base.primitive.clone(),
        diff.primitive.clone(),
        scale.primitive.clone(),
    );

    attach_addcmul_output::<B, C>(base, diff, scale, output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn addcmul5_autodiff<B: AddcmulBackend, C: CheckpointStrategy>(
    base: FloatTensor<Autodiff<B, C>>,
    diff: FloatTensor<Autodiff<B, C>>,
    receptance_scale: FloatTensor<Autodiff<B, C>>,
    weight_decay_scale: FloatTensor<Autodiff<B, C>>,
    key_scale: FloatTensor<Autodiff<B, C>>,
    value_scale: FloatTensor<Autodiff<B, C>>,
    learning_rate_scale: FloatTensor<Autodiff<B, C>>,
) -> Addcmul5OutputPrimitive<Autodiff<B, C>> {
    let Addcmul5Output {
        receptance_input,
        weight_decay_input,
        key_input,
        value_input,
        learning_rate_input,
    } = B::addcmul5(
        base.primitive.clone(),
        diff.primitive.clone(),
        receptance_scale.primitive.clone(),
        weight_decay_scale.primitive.clone(),
        key_scale.primitive.clone(),
        value_scale.primitive.clone(),
        learning_rate_scale.primitive.clone(),
    );

    Addcmul5Output {
        receptance_input: attach_addcmul_output::<B, C>(
            base.clone(),
            diff.clone(),
            receptance_scale,
            receptance_input,
        ),
        weight_decay_input: attach_addcmul_output::<B, C>(
            base.clone(),
            diff.clone(),
            weight_decay_scale,
            weight_decay_input,
        ),
        key_input: attach_addcmul_output::<B, C>(base.clone(), diff.clone(), key_scale, key_input),
        value_input: attach_addcmul_output::<B, C>(
            base.clone(),
            diff.clone(),
            value_scale,
            value_input,
        ),
        learning_rate_input: attach_addcmul_output::<B, C>(
            base,
            diff,
            learning_rate_scale,
            learning_rate_input,
        ),
    }
}
