use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::CubeRuntime;

use crate::kernels::rapid_sample::{
    host::rapid_sample_topk_topp_impl, RapidSampleBackend, RapidSampleOutputPrimitive,
    RapidSamplePenaltyConfig,
};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> RapidSampleBackend
    for CubeBackend<R, F, I, BT>
{
    fn rapid_sample(
        logits: FloatTensor<Self>,
        states: IntTensor<Self>,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        penalties: Option<(FloatTensor<Self>, RapidSamplePenaltyConfig)>,
    ) -> RapidSampleOutputPrimitive<Self> {
        rapid_sample_topk_topp_impl::<R, F, I, BT>(
            logits,
            states,
            temperature,
            top_k,
            top_p,
            penalties,
        )
    }
}
