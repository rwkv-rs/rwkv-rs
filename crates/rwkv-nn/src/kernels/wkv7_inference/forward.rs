use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
use burn::tensor::ops::FloatTensor;
use burn_cubecl::{CubeElement, CubeRuntime};

use crate::kernels::wkv7_inference::{
    Wkv7InferenceBackend, Wkv7InferenceForwardOutput,
    host::wkv7_inference_forward_impl,
};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Wkv7InferenceBackend
    for CubeBackend<R, F, I, BT>
where
    F: CubeElement,
{
    fn wkv7_inference_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> Wkv7InferenceForwardOutput<FloatTensor<Self>> {
        wkv7_inference_forward_impl::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            initial_state,
            context_mask,
        )
    }
}
