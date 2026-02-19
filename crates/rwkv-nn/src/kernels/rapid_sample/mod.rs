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

/// Penalty settings used by repetition-aware sampling.
#[derive(Clone, Copy, Debug)]
pub struct RapidSamplePenaltyConfig {
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub penalty_decay: f32,
}

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
    /// Unified rapid sample entry.
    ///
    /// # Shapes
    /// - `logits`: `[batch_size, vocab_size]`
    /// - `states`: `[batch_size]`
    /// - `penalties.0`: `[batch_size, vocab_size]` when provided, dtype must be `F32`
    fn rapid_sample(
        logits: FloatTensor<Self>,
        states: IntTensor<Self>,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        penalties: Option<(FloatTensor<Self>, RapidSamplePenaltyConfig)>,
    ) -> RapidSampleOutputPrimitive<Self>;
}

pub fn rapid_sample<B: RapidSampleBackend>(
    logits: Tensor<B, 2>,
    states: Tensor<B, 1, Int>,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    penalties: Option<(Tensor<B, 2>, RapidSamplePenaltyConfig)>,
) -> RapidSampleOutputTensor<B> {
    let primitive_penalties =
        penalties.map(|(penalties, config)| (penalties.into_primitive().tensor(), config));

    let out = B::rapid_sample(
        logits.into_primitive().tensor(),
        states.into_primitive(),
        temperature,
        top_k,
        top_p,
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
