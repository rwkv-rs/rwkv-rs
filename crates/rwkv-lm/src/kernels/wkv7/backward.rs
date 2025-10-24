/// Autodiff wrapper approximating the CUDA backward kernel in model.py L46-L68.
use burn::tensor::ops::FloatTensor;
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

use crate::kernels::wkv7::{Wkv7AutodiffBackend, Wkv7Backend};

// CubeCL imports removed as they're not used in this file
/// Implement WKV7Backend for Autodiff-decorated backends

impl<B: Wkv7Backend, C: CheckpointStrategy> Wkv7Backend for Autodiff<B, C> {
    fn wkv7_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: Option<FloatTensor<Self>>,
        chunk_len: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]

        struct Wkv7Backward;

        impl<B: Wkv7Backend> Backward<B, 6> for Wkv7Backward {
            type State = (
                NodeId,
                NodeId,
                NodeId,
                NodeId,
                NodeId,
                NodeId, // inputs
                FloatTensor<B>,
                FloatTensor<B>, // intermediates (s, sa) saved directly
                usize,
                Option<NodeId>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 6>,
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
                ] = ops.parents;

                let grad = grads.consume::<B>(&ops.node);

                let (
                    weight_decay_state,
                    receptance_state,
                    key_state,
                    value_state,
                    removal_state,
                    replacement_state,
                    state_saved,
                    removal_state_saved,
                    chunk_len,
                    initial_state_node,
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

                let state: FloatTensor<B> = state_saved;

                let removal_state_tensor: FloatTensor<B> = removal_state_saved;

                // Compute gradients for all inputs using the backward operation
                let (
                    weight_decay_grad,
                    receptance_grad,
                    key_grad,
                    value_grad,
                    removal_grad,
                    replacement_grad,
                    grad_initial_state,
                ) = B::wkv7_backward(
                    weight_decay,
                    receptance,
                    key,
                    value,
                    removal,
                    replacement,
                    state,
                    removal_state_tensor,
                    grad,
                    chunk_len,
                );

                // Register gradients for tracked variables
                if let Some(node) = node_weight_decay {
                    grads.register::<B>(node.id, weight_decay_grad);
                }

                if let Some(node) = node_receptance {
                    grads.register::<B>(node.id, receptance_grad);
                }

                if let Some(node) = node_key {
                    grads.register::<B>(node.id, key_grad);
                }

                if let Some(node) = node_value {
                    grads.register::<B>(node.id, value_grad);
                }

                if let Some(node) = node_removal {
                    grads.register::<B>(node.id, removal_grad);
                }

                if let Some(node) = node_replacement {
                    grads.register::<B>(node.id, replacement_grad);
                }

                if let Some(node_id) = initial_state_node {
                    grads.register::<B>(node_id, grad_initial_state);
                }
            }
        }

        // Prepare a stateful operation with each variable node
        let backward_op = Wkv7Backward;

        match backward_op
            .prepare::<C>([
                weight_decay.node.clone(),
                receptance.node.clone(),
                key.node.clone(),
                value.node.clone(),
                removal.node.clone(),
                replacement.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // When at least one node is tracked, register backward step
                let weight_decay_state = prep.checkpoint(&weight_decay);

                let receptance_state = prep.checkpoint(&receptance);

                let key_state = prep.checkpoint(&key);

                let value_state = prep.checkpoint(&value);

                let removal_state = prep.checkpoint(&removal);

                let replacement_state = prep.checkpoint(&replacement);

                let initial_state_node = initial_state.as_ref().map(|tensor| tensor.node.id);

                let initial_state_primitive = initial_state.as_ref().map(|s| s.primitive.clone());

                let (state_out, removal_state_out, output) = B::wkv7_forward(
                    weight_decay.primitive.clone(),
                    receptance.primitive.clone(),
                    key.primitive.clone(),
                    value.primitive.clone(),
                    removal.primitive.clone(),
                    replacement.primitive.clone(),
                    initial_state_primitive,
                    chunk_len,
                );

                // Save intermediates directly in the state to avoid checkpointing untracked
                // tensors
                let saved_state = (
                    weight_decay_state,
                    receptance_state,
                    key_state,
                    value_state,
                    removal_state,
                    replacement_state, // inputs
                    state_out.clone(),
                    removal_state_out.clone(), // intermediates
                    chunk_len,
                    initial_state_node,
                );

                let output_tensor = prep.finish(saved_state, output.clone());

                // Return Autodiff wrappers for s and sa; they don't need to be checkpointed
                let state_tensor = Autodiff::<B, C>::from_inner(state_out);

                let removal_state_autodiff = Autodiff::<B, C>::from_inner(removal_state_out);

                // Match trait contract: (s, sa, y)
                (state_tensor, removal_state_autodiff, output_tensor)
            },
            OpsKind::UnTracked(prep) => {
                let initial_state_primitive = initial_state.map(|s| s.primitive);

                let (state_out, removal_state_out, output) = B::wkv7_forward(
                    weight_decay.primitive,
                    receptance.primitive,
                    key.primitive,
                    value.primitive,
                    removal.primitive,
                    replacement.primitive,
                    initial_state_primitive,
                    chunk_len,
                );

                let output_tensor = prep.finish(output);

                let state_tensor = Autodiff::<B, C>::from_inner(state_out);

                let removal_state_autodiff = Autodiff::<B, C>::from_inner(removal_state_out);

                (state_tensor, removal_state_autodiff, output_tensor)
            },
        }
    }

    fn wkv7_backward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        state: FloatTensor<Self>,
        removal_state: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        chunk_len: usize,
    ) -> (
        FloatTensor<Self>, // dw
        FloatTensor<Self>, // dq
        FloatTensor<Self>, // dk
        FloatTensor<Self>, // dv
        FloatTensor<Self>, // da
        FloatTensor<Self>, // db
        FloatTensor<Self>, // dh0
    ) {
        // For autodiff backend, delegate to inner backend
        let (dw, dq, dk, dv, da, db, dh0) = B::wkv7_backward(
            Autodiff::<B, C>::inner(weight_decay),
            Autodiff::<B, C>::inner(receptance),
            Autodiff::<B, C>::inner(key),
            Autodiff::<B, C>::inner(value),
            Autodiff::<B, C>::inner(removal),
            Autodiff::<B, C>::inner(replacement),
            Autodiff::<B, C>::inner(state),
            Autodiff::<B, C>::inner(removal_state),
            Autodiff::<B, C>::inner(output_grad),
            chunk_len,
        );

        (
            Autodiff::<B, C>::from_inner(dw),
            Autodiff::<B, C>::from_inner(dq),
            Autodiff::<B, C>::from_inner(dk),
            Autodiff::<B, C>::from_inner(dv),
            Autodiff::<B, C>::from_inner(da),
            Autodiff::<B, C>::from_inner(db),
            Autodiff::<B, C>::from_inner(dh0),
        )
    }
}

/// Implement Wkv7AutodiffBackend for any Autodiff-decorated backend that
/// implements Wkv7Backend

impl<B: Wkv7Backend, C: CheckpointStrategy> Wkv7AutodiffBackend for Autodiff<B, C> {}

#[cfg(test)]

mod tests {

    use burn::tensor::Tensor;

    use super::*;
    use crate::utils::test_tools::*;

    #[test]

    fn test_wkv7_backward() {
        let weight_decay = load_expected_f32::<4>("wkv7_kernel_backward_saved_w");

        let receptance = load_expected_f32::<4>("wkv7_kernel_backward_saved_q");

        let key = load_expected_f32::<4>("wkv7_kernel_backward_saved_k");

        let value = load_expected_f32::<4>("wkv7_kernel_backward_saved_v");

        let removal = load_expected_f32::<4>("wkv7_kernel_backward_saved_z");

        let replacement = load_expected_f32::<4>("wkv7_kernel_backward_saved_b");

        let state = load_expected_f32::<5>("wkv7_kernel_backward_saved_s");

        let removal_state = load_expected_f32::<4>("wkv7_kernel_backward_saved_sa");

        let output_grad = load_expected_f32::<4>("wkv7_kernel_backward_input_dy");

        let chunk_len = 16;

        let (
            weight_decay_grad,
            receptance_grad,
            key_grad,
            value_grad,
            removal_grad,
            replacement_grad,
            _initial_state_grad,
        ) = TestAutodiffBackend::wkv7_backward(
            weight_decay.into_primitive().tensor(),
            receptance.into_primitive().tensor(),
            key.into_primitive().tensor(),
            value.into_primitive().tensor(),
            removal.into_primitive().tensor(),
            replacement.into_primitive().tensor(),
            state.into_primitive().tensor(),
            removal_state.into_primitive().tensor(),
            output_grad.into_primitive().tensor(),
            chunk_len,
        );

        let weight_decay_grad_tensor = Tensor::<TestAutodiffBackend, 4>::from_primitive(
            burn::tensor::TensorPrimitive::Float(weight_decay_grad),
        );

        let receptance_grad_tensor = Tensor::<TestAutodiffBackend, 4>::from_primitive(
            burn::tensor::TensorPrimitive::Float(receptance_grad),
        );

        let key_grad_tensor = Tensor::<TestAutodiffBackend, 4>::from_primitive(
            burn::tensor::TensorPrimitive::Float(key_grad),
        );

        let value_grad_tensor = Tensor::<TestAutodiffBackend, 4>::from_primitive(
            burn::tensor::TensorPrimitive::Float(value_grad),
        );

        let removal_grad_tensor = Tensor::<TestAutodiffBackend, 4>::from_primitive(
            burn::tensor::TensorPrimitive::Float(removal_grad),
        );

        let replacement_grad_tensor = Tensor::<TestAutodiffBackend, 4>::from_primitive(
            burn::tensor::TensorPrimitive::Float(replacement_grad),
        );

        // 加载期望结果 - 注意变量映射关系
        let expected_weight_decay_grad = load_expected_f32::<4>("wkv7_kernel_backward_dw");

        let expected_receptance_grad = load_expected_f32::<4>("wkv7_kernel_backward_dq");

        let expected_key_grad = load_expected_f32::<4>("wkv7_kernel_backward_dk");

        let expected_value_grad = load_expected_f32::<4>("wkv7_kernel_backward_dv");

        let expected_removal_grad = load_expected_f32::<4>("wkv7_kernel_backward_dz"); // CUDA 的 dz 对应 removal
        let expected_replacement_grad = load_expected_f32::<4>("wkv7_kernel_backward_db"); // CUDA 的 da 对应 replacement
        let actual_vec = vec![
            weight_decay_grad_tensor,
            receptance_grad_tensor,
            key_grad_tensor,
            value_grad_tensor,
            removal_grad_tensor,
            replacement_grad_tensor,
        ];

        let expected_vec = vec![
            expected_weight_decay_grad,
            expected_receptance_grad,
            expected_key_grad,
            expected_value_grad,
            expected_removal_grad,
            expected_replacement_grad,
        ];

        let module_names = vec![
            "wkv7_kernel_backward_dw".to_string(),
            "wkv7_kernel_backward_dq".to_string(),
            "wkv7_kernel_backward_dk".to_string(),
            "wkv7_kernel_backward_dv".to_string(),
            "wkv7_kernel_backward_da".to_string(),
            "wkv7_kernel_backward_db".to_string(),
        ];

        assert_closeness_multi(
            actual_vec,
            expected_vec,
            module_names,
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
    }
}
