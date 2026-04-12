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
//! - an inference-oriented `batch_ids` entry point lets the kernel update the full RNG /
//!   penalty state in place without a `[batch, vocab]` gather + scatter around every decode step.
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
        Int,
        Tensor,
        TensorPrimitive,
        ops::{FloatTensor, IntTensor},
    },
};

/// Output of a `rapid_sample` invocation.
#[derive(Clone, Debug)]
pub struct RapidSampleOutput<FT, IT> {
    /// Sampled token ids, shape `[batch_size]`.
    pub token_ids: IT,
    /// Updated RNG states, shape `[batch_size]`.
    pub states: IT,
    /// Post-sampling probabilities, shape `[batch_size, vocab_size]`.
    pub probs: FT,
    /// Updated penalties, shape `[batch_size, vocab_size]` when enabled.
    pub penalties: Option<FT>,
}

/// `rapid_sample` output expressed with standard tensor wrappers.
pub type RapidSampleOutputTensor<B> = RapidSampleOutput<Tensor<B, 2>, Tensor<B, 1, Int>>;
/// `rapid_sample` output expressed with backend primitives.
pub type RapidSampleOutputPrimitive<B> = RapidSampleOutput<FloatTensor<B>, IntTensor<B>>;

/// Backend hook for device-side rapid sampling.
#[allow(clippy::too_many_arguments)]
pub trait RapidSampleBackend: Backend {
    /// Executes one rapid-sampling step for each active batch lane and updates RNG / penalty
    /// state in place.
    ///
    /// Callers are expected to pass pre-normalized sampling parameters so the kernel path does
    /// not repeat cheap host-side setup on every decode step.
    ///
    /// # Shapes
    /// - `logits`: `[batch_size, vocab_size]`
    /// - `states`: `[batch_size]`
    /// - `inv_temperatures`: `[batch_size]` (pre-computed `1.0 / temperature`)
    /// - `top_ks`: `[batch_size]` (pre-normalized via `normalize_topk_topp`)
    /// - `top_ps`: `[batch_size]` (pre-normalized via `normalize_topk_topp`)
    /// - `batch_ids`: `[batch_size]` active-row -> full-state-slot mapping
    /// - penalties tuple:
    ///   - `.0`: `[full_batch_size, vocab_size]` penalty state, dtype `F32`
    ///   - `.1`: `[batch_size]` presence_penalty
    ///   - `.2`: `[batch_size]` repetition_penalty
    ///   - `.3`: `[batch_size]` penalty_decay
    fn rapid_sample(
        logits: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        states: IntTensor<Self>,
        inv_temperatures: FloatTensor<Self>,
        top_ks: IntTensor<Self>,
        top_ps: FloatTensor<Self>,
        penalties: (
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
            FloatTensor<Self>,
        ),
    ) -> RapidSampleOutputPrimitive<Self>;
}

#[cfg_attr(
    feature = "trace",
    tracing::instrument(name = "rwkv.infer.executor.rapid_sample", skip_all)
)]
/// Invokes device-side rapid sampling and converts backend primitives back into regular tensors.
///
/// # Panics
/// Panics if the backend implementation detects that tensor shapes or state mappings violate the
/// backend contract.
pub fn rapid_sample<B: RapidSampleBackend>(
    logits: Tensor<B, 2>,
    batch_ids: Tensor<B, 1, Int>,
    rng: Tensor<B, 1, Int>,
    inv_temperatures: Tensor<B, 1>,
    top_ks: Tensor<B, 1, Int>,
    top_ps: Tensor<B, 1>,
    penalties: (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>),
) -> RapidSampleOutputTensor<B> {
    let (pen, pp, rp, pd) = penalties;
    let primitive_penalties = (
        pen.into_primitive().tensor(),
        pp.into_primitive().tensor(),
        rp.into_primitive().tensor(),
        pd.into_primitive().tensor(),
    );

    let out = B::rapid_sample(
        logits.into_primitive().tensor(),
        batch_ids.into_primitive(),
        rng.into_primitive(),
        inv_temperatures.into_primitive().tensor(),
        top_ks.into_primitive(),
        top_ps.into_primitive().tensor(),
        primitive_penalties,
    );

    RapidSampleOutput {
        token_ids: Tensor::from_primitive(out.token_ids),
        states: Tensor::from_primitive(out.states),
        probs: Tensor::from_primitive(TensorPrimitive::Float(out.probs)),
        penalties: out
            .penalties
            .map(|penalties| Tensor::from_primitive(TensorPrimitive::Float(penalties))),
    }
}
