use burn::tensor::ops::FloatTensor;
use burn::tensor::TensorMetadata;
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
    tensor::backend::AutodiffBackend,
};

use crate::kernels::wkv7_statepass::{Wkv7StatePassAutodiffBackend, Wkv7StatePassBackend};

impl<B: Wkv7StatePassBackend, C: CheckpointStrategy> Wkv7StatePassBackend for Autodiff<B, C> {
    fn wkv7_statepass_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        chunk_len: usize,
    ) -> crate::kernels::wkv7_common::Wkv7StatePassForwardOutput<FloatTensor<Self>> {
        #[derive(Debug)]
        struct Wkv7Backward;

        impl<B: Wkv7StatePassBackend> Backward<B, 7> for Wkv7Backward {
            type State = (
                NodeId,
                NodeId,
                NodeId,
                NodeId,
                NodeId,
                NodeId,
                NodeId,
                FloatTensor<B>,
                FloatTensor<B>,
                usize,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_weight_decay,
                    node_receptance,
                    node_key,
                    node_value,
                    node_removal,
                    node_replacement,
                    node_initial_state,
                ] = ops.parents;

                let grad = grads.consume::<B>(&ops.node);

                let (
                    weight_decay_state,
                    receptance_state,
                    key_state,
                    value_state,
                    removal_state,
                    replacement_state,
                    initial_state_state,
                    state_saved,
                    removal_state_saved,
                    chunk_len,
                ) = ops.state;

                let weight_decay: FloatTensor<B> =
                    checkpointer.retrieve_node_output(weight_decay_state);
                let receptance: FloatTensor<B> =
                    checkpointer.retrieve_node_output(receptance_state);
                let key: FloatTensor<B> = checkpointer.retrieve_node_output(key_state);
                let value: FloatTensor<B> = checkpointer.retrieve_node_output(value_state);
                let removal: FloatTensor<B> = checkpointer.retrieve_node_output(removal_state);
                let replacement: FloatTensor<B> =
                    checkpointer.retrieve_node_output(replacement_state);
                let initial_state: FloatTensor<B> =
                    checkpointer.retrieve_node_output(initial_state_state);

                let state: FloatTensor<B> = state_saved;
                let removal_state_tensor: FloatTensor<B> = removal_state_saved;

                let final_state_grad = B::float_zeros(
                    initial_state.shape(),
                    &B::float_device(&initial_state),
                    initial_state.dtype().into(),
                );

                let grads_out = B::wkv7_statepass_backward(
                    weight_decay,
                    receptance,
                    key,
                    value,
                    removal,
                    replacement,
                    state,
                    removal_state_tensor,
                    grad,
                    final_state_grad,
                    chunk_len,
                );

                if let Some(node) = node_weight_decay {
                    grads.register::<B>(node.id, grads_out.weight_decay_grad);
                }
                if let Some(node) = node_receptance {
                    grads.register::<B>(node.id, grads_out.receptance_grad);
                }
                if let Some(node) = node_key {
                    grads.register::<B>(node.id, grads_out.key_grad);
                }
                if let Some(node) = node_value {
                    grads.register::<B>(node.id, grads_out.value_grad);
                }
                if let Some(node) = node_removal {
                    grads.register::<B>(node.id, grads_out.removal_grad);
                }
                if let Some(node) = node_replacement {
                    grads.register::<B>(node.id, grads_out.replacement_grad);
                }
                if let Some(node) = node_initial_state {
                    grads.register::<B>(node.id, grads_out.initial_state_grad);
                }

            }
        }

        let backward_op = Wkv7Backward;

        match backward_op
            .prepare::<C>([
                weight_decay.node.clone(),
                receptance.node.clone(),
                key.node.clone(),
                value.node.clone(),
                removal.node.clone(),
                replacement.node.clone(),
                initial_state.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let weight_decay_state = prep.checkpoint(&weight_decay);
                let receptance_state = prep.checkpoint(&receptance);
                let key_state = prep.checkpoint(&key);
                let value_state = prep.checkpoint(&value);
                let removal_state = prep.checkpoint(&removal);
                let replacement_state = prep.checkpoint(&replacement);
                let initial_state_state = prep.checkpoint(&initial_state);

                let output = B::wkv7_statepass_forward(
                    weight_decay.primitive.clone(),
                    receptance.primitive.clone(),
                    key.primitive.clone(),
                    value.primitive.clone(),
                    removal.primitive.clone(),
                    replacement.primitive.clone(),
                    initial_state.primitive.clone(),
                    chunk_len,
                );

                let saved_state = (
                    weight_decay_state,
                    receptance_state,
                    key_state,
                    value_state,
                    removal_state,
                    replacement_state,
                    initial_state_state,
                    output.state.clone(),
                    output.removal_state.clone(),
                    chunk_len,
                );

                let output_tensor = prep.finish(saved_state, output.output.clone());

                let state_tensor = Autodiff::<B, C>::from_inner(output.state);
                let removal_state_tensor = Autodiff::<B, C>::from_inner(output.removal_state);
                let final_state_tensor = Autodiff::<B, C>::from_inner(output.final_state);

                crate::kernels::wkv7_common::Wkv7StatePassForwardOutput {
                    state: state_tensor,
                    removal_state: removal_state_tensor,
                    output: output_tensor,
                    final_state: final_state_tensor,
                }
            }
            OpsKind::UnTracked(prep) => {
                let output = B::wkv7_statepass_forward(
                    weight_decay.primitive,
                    receptance.primitive,
                    key.primitive,
                    value.primitive,
                    removal.primitive,
                    replacement.primitive,
                    initial_state.primitive,
                    chunk_len,
                );

                let output_tensor = prep.finish(output.output);

                let state_tensor = Autodiff::<B, C>::from_inner(output.state);
                let removal_state_tensor = Autodiff::<B, C>::from_inner(output.removal_state);
                let final_state_tensor = Autodiff::<B, C>::from_inner(output.final_state);

                crate::kernels::wkv7_common::Wkv7StatePassForwardOutput {
                    state: state_tensor,
                    removal_state: removal_state_tensor,
                    output: output_tensor,
                    final_state: final_state_tensor,
                }
            }
        }
    }

    fn wkv7_statepass_backward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        state: FloatTensor<Self>,
        removal_state: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        final_state_grad: FloatTensor<Self>,
        chunk_len: usize,
    ) -> crate::kernels::wkv7_common::Wkv7StateBackwardOutput<FloatTensor<Self>> {
        let grads_out = B::wkv7_statepass_backward(
            Autodiff::<B, C>::inner(weight_decay),
            Autodiff::<B, C>::inner(receptance),
            Autodiff::<B, C>::inner(key),
            Autodiff::<B, C>::inner(value),
            Autodiff::<B, C>::inner(removal),
            Autodiff::<B, C>::inner(replacement),
            Autodiff::<B, C>::inner(state),
            Autodiff::<B, C>::inner(removal_state),
            Autodiff::<B, C>::inner(output_grad),
            Autodiff::<B, C>::inner(final_state_grad),
            chunk_len,
        );

        crate::kernels::wkv7_common::Wkv7StateBackwardOutput {
            weight_decay_grad: Autodiff::<B, C>::from_inner(grads_out.weight_decay_grad),
            receptance_grad: Autodiff::<B, C>::from_inner(grads_out.receptance_grad),
            key_grad: Autodiff::<B, C>::from_inner(grads_out.key_grad),
            value_grad: Autodiff::<B, C>::from_inner(grads_out.value_grad),
            removal_grad: Autodiff::<B, C>::from_inner(grads_out.removal_grad),
            replacement_grad: Autodiff::<B, C>::from_inner(grads_out.replacement_grad),
            initial_state_grad: Autodiff::<B, C>::from_inner(grads_out.initial_state_grad),
        }
    }
}

impl<B: Wkv7StatePassBackend, C: CheckpointStrategy> Wkv7StatePassAutodiffBackend
    for Autodiff<B, C>
{
}
