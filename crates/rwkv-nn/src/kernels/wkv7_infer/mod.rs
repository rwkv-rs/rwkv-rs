mod forward;
mod host;
mod kernel;

use burn::{
    backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy},
    prelude::{Backend, Int},
    tensor::{
        Tensor,
        TensorPrimitive,
        backend::AutodiffBackend,
        ops::{FloatTensor, IntTensor},
    },
};

/// Forward output for inference-only WKV7 kernel.
///
/// - `output`: [active_batch_size, context_length, num_heads, head_size]
/// - `final_state`: [full_batch_size, num_heads, head_size, head_size]
#[derive(Clone, Debug)]
pub struct Wkv7InferForwardOutput<T> {
    pub output: T,
    pub final_state: T,
}

pub type Wkv7InferForwardOutputTensor<B> = Wkv7InferForwardOutput<Tensor<B, 4>>;
pub type Wkv7InferForwardOutputPrimitive<B> = Wkv7InferForwardOutput<FloatTensor<B>>;

#[allow(clippy::too_many_arguments)]
pub trait Wkv7InferBackend: Backend {
    /// Inference forward.
    ///
    /// - `context_mask`: [batch_size, context_length] with values 0/1.
    ///   When 0, the corresponding timestep is treated as padding and must be a strict no-op
    ///   for the internal state.
    /// - `elapsed_t`: [full_batch_size] full-slot counters of previously consumed effective
    ///   (non-masked) tokens used by the FP16 deterministic dither path. `batch_ids` maps active
    ///   rows onto these counters, and masked timesteps must not consume dither positions.
    fn wkv7_infer_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        initial_state: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
        elapsed_t: IntTensor<Self>,
    ) -> Wkv7InferForwardOutputPrimitive<Self>;
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    feature = "trace",
    tracing::instrument(name = "rwkv.infer.model.wkv7", skip_all)
)]
pub fn wkv7_infer_forward<B: Wkv7InferBackend>(
    weight_decay: Tensor<B, 4>,
    receptance: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    removal: Tensor<B, 4>,
    replacement: Tensor<B, 4>,
    batch_ids: Tensor<B, 1, Int>,
    initial_state: Tensor<B, 4>,
    context_mask: Tensor<B, 2>,
    elapsed_t: Tensor<B, 1, Int>,
) -> Wkv7InferForwardOutputTensor<B> {
    let output = B::wkv7_infer_forward(
        weight_decay.into_primitive().tensor(),
        receptance.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        removal.into_primitive().tensor(),
        replacement.into_primitive().tensor(),
        batch_ids.into_primitive(),
        initial_state.into_primitive().tensor(),
        context_mask.into_primitive().tensor(),
        elapsed_t.into_primitive(),
    );

    Wkv7InferForwardOutput {
        output: Tensor::from_primitive(TensorPrimitive::Float(output.output)),
        final_state: Tensor::from_primitive(TensorPrimitive::Float(output.final_state)),
    }
}

impl<B: Wkv7InferBackend, C: CheckpointStrategy> Wkv7InferBackend for Autodiff<B, C> {
    fn wkv7_infer_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        batch_ids: IntTensor<Self>,
        initial_state: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
        elapsed_t: IntTensor<Self>,
    ) -> Wkv7InferForwardOutput<FloatTensor<Self>> {
        let output = B::wkv7_infer_forward(
            weight_decay.primitive,
            receptance.primitive,
            key.primitive,
            value.primitive,
            removal.primitive,
            replacement.primitive,
            batch_ids,
            initial_state.primitive,
            context_mask.primitive,
            elapsed_t,
        );

        Wkv7InferForwardOutput {
            output: Autodiff::<B, C>::from_inner(output.output),
            final_state: Autodiff::<B, C>::from_inner(output.final_state),
        }
    }
}

#[cfg(test)]
mod tests {
    const W_SCALE: f32 = -0.6065306597;
    const TWO_TO_NEG_41: f32 = 4.547473508864641e-13;
    const ROTATOR1: u32 = 2654435769u32;

    #[derive(Clone, Copy)]
    struct ScalarStep {
        weight_decay: f32,
        receptance: f32,
        key: f32,
        value: f32,
        removal: f32,
        replacement: f32,
    }

    fn deterministic_dither(seed: u32) -> f32 {
        let mixed = seed.wrapping_mul(ROTATOR1) as i32;
        (mixed as f32) * TWO_TO_NEG_41
    }

    fn run_scalar_forward(
        steps: &[ScalarStep],
        mask: &[bool],
        elapsed_t: u32,
        count_masked_steps_in_seed: bool,
    ) -> (Vec<f32>, f32) {
        assert_eq!(steps.len(), mask.len());

        let mut state = 0.0f32;
        let mut outputs = Vec::with_capacity(steps.len());
        let mut valid_t = 0u32;

        for (t, (&step, &is_valid)) in steps.iter().zip(mask.iter()).enumerate() {
            if !is_valid {
                outputs.push(0.0);
                continue;
            }

            let weight_sigmoid = 1.0 / (1.0 + (-step.weight_decay).exp());
            let seed = if count_masked_steps_in_seed {
                elapsed_t + t as u32
            } else {
                elapsed_t + valid_t
            };
            let weight_decay = (W_SCALE * weight_sigmoid).exp() + deterministic_dither(seed);
            let removal_state = step.removal * state;

            state = state * weight_decay + removal_state * step.replacement + step.key * step.value;
            outputs.push(state * step.receptance);
            valid_t += 1;
        }

        (outputs, state)
    }

    fn assert_close(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (&lhs, &rhs) in lhs.iter().zip(rhs.iter()) {
            assert!(
                (lhs - rhs).abs() <= tolerance,
                "lhs={lhs} rhs={rhs} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn masked_dither_seed_matches_unpadded_execution() {
        let elapsed_t = 17u32;
        let unpadded_steps = [
            ScalarStep {
                weight_decay: -1.2,
                receptance: 0.7,
                key: 0.3,
                value: 1.1,
                removal: 0.2,
                replacement: 0.9,
            },
            ScalarStep {
                weight_decay: 0.4,
                receptance: 1.2,
                key: -0.5,
                value: 0.8,
                removal: -0.3,
                replacement: 1.05,
            },
        ];
        let padded_steps = [
            ScalarStep {
                weight_decay: 3.0,
                receptance: 2.0,
                key: 4.0,
                value: 5.0,
                removal: 6.0,
                replacement: 7.0,
            },
            unpadded_steps[0],
            unpadded_steps[1],
        ];

        let (unpadded_outputs, unpadded_state) =
            run_scalar_forward(&unpadded_steps, &[true, true], elapsed_t, false);
        let (padded_outputs, padded_state) =
            run_scalar_forward(&padded_steps, &[false, true, true], elapsed_t, false);

        assert_eq!(padded_outputs[0], 0.0);
        assert_close(&padded_outputs[1..], &unpadded_outputs, 1e-7);
        assert!((padded_state - unpadded_state).abs() <= 1e-7);
    }

    #[test]
    fn counting_masked_steps_in_seed_shifts_left_padded_prefill() {
        let elapsed_t = 17u32;
        let steps = [
            ScalarStep {
                weight_decay: 3.0,
                receptance: 2.0,
                key: 4.0,
                value: 5.0,
                removal: 6.0,
                replacement: 7.0,
            },
            ScalarStep {
                weight_decay: -1.2,
                receptance: 0.7,
                key: 0.3,
                value: 1.1,
                removal: 0.2,
                replacement: 0.9,
            },
            ScalarStep {
                weight_decay: 0.4,
                receptance: 1.2,
                key: -0.5,
                value: 0.8,
                removal: -0.3,
                replacement: 1.05,
            },
        ];

        let (_, correct_state) = run_scalar_forward(&steps, &[false, true, true], elapsed_t, false);
        let (_, shifted_state) = run_scalar_forward(&steps, &[false, true, true], elapsed_t, true);

        assert!(
            (correct_state - shifted_state).abs() > 1e-9,
            "left padding should change the dither seed if masked steps are counted"
        );
    }
}
