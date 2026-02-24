//! Rapid token sampling kernels.
//!
//! Original `rapid-sampling` CUDA design preserved:
//! - monotone block scan/reduce for threshold statistics;
//! - bit-space quaternary threshold search (`float <-> u32` reinterpret);
//! - top-k/top-p threshold compensation;
//! - two-stage sampling (`tile` first, then token in tile).
//!
//! CubeCL adaptations:
//! - `Line<f32>` vectorized loads/stores (`VEC = 4`, CUDA `float4` equivalent);
//! - plane-shuffle + shared memory reductions/scans;
//! - RNG state uses lightweight LCG (`u32`) instead of CUDA Philox.
//!
//! Equivalence target:
//! - algorithmic equivalence and statistically consistent sampling behavior;
//! - not bitwise-identical to CUDA RNG paths.

mod forward;
mod host;
mod kernel;

use burn::{
    prelude::Backend,
    tensor::{
        Int, Tensor, TensorPrimitive,
        ops::{FloatTensor, IntTensor},
    },
};

pub use host::normalize_topk_topp;

/// Unified rapid-sample output.
#[derive(Clone, Debug)]
pub struct RapidSampleOutput<FT, IT> {
    /// Sampled token ids, shape `[batch_size]`.
    pub token_ids: IT,
    /// Updated RNG states, shape `[batch_size]`.
    pub states: IT,
    /// Updated penalties, shape `[batch_size, vocab_size]` when enabled.
    pub penalties: Option<FT>,
}

pub type RapidSampleOutputTensor<B> = RapidSampleOutput<Tensor<B, 2>, Tensor<B, 1, Int>>;
pub type RapidSampleOutputPrimitive<B> = RapidSampleOutput<FloatTensor<B>, IntTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait RapidSampleBackend: Backend {
    /// Per-batch rapid sample.
    ///
    /// All sampling parameters are per-batch tensors (pre-normalized by caller).
    ///
    /// # Shapes
    /// - `logits`: `[batch_size, vocab_size]`
    /// - `states`: `[batch_size]`
    /// - `inv_temperatures`: `[batch_size]` (pre-computed `1.0 / temperature`)
    /// - `top_ks`: `[batch_size]` (pre-normalized via `normalize_topk_topp`)
    /// - `top_ps`: `[batch_size]` (pre-normalized via `normalize_topk_topp`)
    /// - penalties tuple when provided:
    ///   - `.0`: `[batch_size, vocab_size]` penalty state, dtype `F32`
    ///   - `.1`: `[batch_size]` presence_penalty
    ///   - `.2`: `[batch_size]` repetition_penalty
    ///   - `.3`: `[batch_size]` penalty_decay
    fn rapid_sample(
        logits: FloatTensor<Self>,
        states: IntTensor<Self>,
        inv_temperatures: FloatTensor<Self>,
        top_ks: IntTensor<Self>,
        top_ps: FloatTensor<Self>,
        penalties: Option<(
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
        )>,
    ) -> RapidSampleOutputPrimitive<Self>;
}

pub fn rapid_sample<B: RapidSampleBackend>(
    logits: Tensor<B, 2>,
    states: Tensor<B, 1, Int>,
    inv_temperatures: Tensor<B, 1>,
    top_ks: Tensor<B, 1, Int>,
    top_ps: Tensor<B, 1>,
    penalties: Option<(Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>)>,
) -> RapidSampleOutputTensor<B> {
    let primitive_penalties = penalties.map(|(pen, pp, rp, pd)| {
        (
            pen.into_primitive().tensor(),
            pp.into_primitive().tensor(),
            rp.into_primitive().tensor(),
            pd.into_primitive().tensor(),
        )
    });

    let out = B::rapid_sample(
        logits.into_primitive().tensor(),
        states.into_primitive(),
        inv_temperatures.into_primitive().tensor(),
        top_ks.into_primitive(),
        top_ps.into_primitive().tensor(),
        primitive_penalties,
    );

    RapidSampleOutput {
        token_ids: Tensor::from_primitive(out.token_ids),
        states: Tensor::from_primitive(out.states),
        penalties: out
            .penalties
            .map(|penalties| Tensor::from_primitive(TensorPrimitive::Float(penalties))),
    }
}
