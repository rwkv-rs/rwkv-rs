use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use rwkv::config::raw::infer::GenerationConfig;
use rwkv::config::validated::model::FinalModelConfig;
use rwkv::custom::Tensor;
use rwkv::custom::cubecl::device::DeviceId;
use rwkv::custom::module::Module;
use rwkv::custom::prelude::{Backend, DeviceOps, Int, TensorData};
use rwkv::custom::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rwkv::custom::tensor::DType;
use rwkv::infer::engine::{EngineRuntime, EngineRuntimeConfig, ModelForward};
use rwkv::infer::service::builder::build_model_group_engines;
use rwkv::infer::service::{ModelEngineFactory, ModelRuntimeGroup};
use rwkv::infer::{Error, Result, SamplingConfig};

use rwkv::nn::kernels::rapid_sample::{RapidSampleBackend, normalize_topk_topp, rapid_sample};
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;

use crate::model::{AutoRegressiveModel, AutoRegressiveModelConfig, UnembedMode};

rwkv::custom_mode!();

/// Pre-allocated recurrent state for a batch of RWKV inference slots.
struct BatchState<B: Backend> {
    embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>>,    // [num_cells] of [max_batch, embedded_dim]
    wkv_state: Vec<Tensor<B, 4>>,                            // [num_cells] of [max_batch, heads, hs, hs]
    embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>>, // [num_cells] of [max_batch, embedded_dim]
    rng_states: Tensor<B, 1, Int>,                           // [max_batch]
    penalties: Tensor<B, 2>,                                  // [max_batch, vocab_size]
}

/// Gather state slices for active batch positions into contiguous tensors of shape [active_count, ...].
fn gather_state<B: Backend>(
    state: &BatchState<B>,
    active_indices: &[usize],
) -> (Vec<Tensor<B, 2>>, Vec<Tensor<B, 4>>, Vec<Tensor<B, 2>>) {
    let num_cells = state.embedded_token_shift_for_time_mix.len();
    let mut ts_time = Vec::with_capacity(num_cells);
    let mut wkv = Vec::with_capacity(num_cells);
    let mut ts_chan = Vec::with_capacity(num_cells);

    for cell_idx in 0..num_cells {
        let slices_time: Vec<Tensor<B, 2>> = active_indices
            .iter()
            .map(|&bi| {
                state.embedded_token_shift_for_time_mix[cell_idx]
                    .clone()
                    .slice([bi..bi + 1])
            })
            .collect();
        ts_time.push(Tensor::cat(slices_time, 0));

        let slices_wkv: Vec<Tensor<B, 4>> = active_indices
            .iter()
            .map(|&bi| state.wkv_state[cell_idx].clone().slice([bi..bi + 1]))
            .collect();
        wkv.push(Tensor::cat(slices_wkv, 0));

        let slices_chan: Vec<Tensor<B, 2>> = active_indices
            .iter()
            .map(|&bi| {
                state.embedded_token_shift_for_channel_mix[cell_idx]
                    .clone()
                    .slice([bi..bi + 1])
            })
            .collect();
        ts_chan.push(Tensor::cat(slices_chan, 0));
    }

    (ts_time, wkv, ts_chan)
}

/// Scatter updated state slices back into the full batch state.
fn scatter_state<B: Backend>(
    state: &mut BatchState<B>,
    active_indices: &[usize],
    ts_time: Vec<Tensor<B, 2>>,
    wkv: Vec<Tensor<B, 4>>,
    ts_chan: Vec<Tensor<B, 2>>,
) {
    let num_cells = state.embedded_token_shift_for_time_mix.len();
    for cell_idx in 0..num_cells {
        for (i, &bi) in active_indices.iter().enumerate() {
            let row = ts_time[cell_idx].clone().slice([i..i + 1]);
            state.embedded_token_shift_for_time_mix[cell_idx] = state
                .embedded_token_shift_for_time_mix[cell_idx]
                .clone()
                .slice_assign([bi..bi + 1], row);
        }
        for (i, &bi) in active_indices.iter().enumerate() {
            let row = wkv[cell_idx].clone().slice([i..i + 1]);
            state.wkv_state[cell_idx] = state.wkv_state[cell_idx]
                .clone()
                .slice_assign([bi..bi + 1], row);
        }
        for (i, &bi) in active_indices.iter().enumerate() {
            let row = ts_chan[cell_idx].clone().slice([i..i + 1]);
            state.embedded_token_shift_for_channel_mix[cell_idx] = state
                .embedded_token_shift_for_channel_mix[cell_idx]
                .clone()
                .slice_assign([bi..bi + 1], row);
        }
    }
}

/// Run model forward on only the active batch positions.
/// Returns logits [active_count, vocab] when unembed_mode != Skip, else None.
#[cfg_attr(feature = "trace", tracing::instrument(name = "rwkv.infer.executor.forward", skip_all))]
fn forward_active<B: Backend + Wkv7Backend>(
    model: &AutoRegressiveModel<B>,
    state: &mut BatchState<B>,
    batch_positions: &[(usize, &[i32], &[u8])],
    unembed_mode: UnembedMode,
    device: &B::Device,
) -> Result<Option<Tensor<B, 2>>> {
    if batch_positions.is_empty() {
        return Ok(None);
    }

    let active_count = batch_positions.len();
    let context_len = batch_positions[0].1.len();

    for (_, tids, cm) in batch_positions {
        if tids.len() != context_len || cm.len() != context_len {
            return Err(Error::BadRequest(
                "forward: inconsistent token_ids/context_mask length".to_string(),
            ));
        }
    }

    let active_indices: Vec<usize> = batch_positions.iter().map(|(bi, _, _)| *bi).collect();

    // Build [active_count, context_len] tensors from only the active positions.
    let mut flat_tokens = Vec::with_capacity(active_count * context_len);
    let mut flat_mask = Vec::with_capacity(active_count * context_len);
    for (_, token_ids, context_mask) in batch_positions {
        flat_tokens.extend_from_slice(token_ids);
        flat_mask.extend(context_mask.iter().map(|&m| if m == 0 { 0.0f32 } else { 1.0f32 }));
    }

    let tokens = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat_tokens, [active_count, context_len]),
        device,
    );
    let context_mask = Tensor::<B, 2>::from_data(
        TensorData::new(flat_mask, [active_count, context_len]),
        device,
    );

    // Gather state for active positions.
    let (mut ts_time, mut wkv, mut ts_chan) = {
        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.gather_state");
        gather_state(state, &active_indices)
    };

    // Forward.
    #[cfg(feature = "nsys")]
    let _nvtx = nvtx::range!("rwkv.infer.executor.forward");
    let logits_3d = model.infer(
        tokens,
        Some(context_mask),
        &mut ts_time,
        &mut wkv,
        &mut ts_chan,
        unembed_mode,
    );

    // Scatter state back.
    {
        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.scatter_state");
        scatter_state(state, &active_indices, ts_time, wkv, ts_chan);
    }

    // logits_3d: Option<[active_count, 1_or_ctx, vocab]> → squeeze to [active_count, vocab]
    Ok(logits_3d.map(|t| {
        let [b, _ctx, v] = t.dims();
        if _ctx == 1 {
            t.reshape([b, v])
        } else {
            // Full mode: take last token per position for sampling.
            t.slice([0..b, (_ctx - 1).._ctx]).reshape([b, v])
        }
    }))
}

const MIN_TEMP: f32 = 0.001;
const MAX_TEMP: f32 = 1000.0;

/// Sample tokens from logits for active positions, with per-position sampling configs.
#[cfg_attr(feature = "trace", tracing::instrument(name = "rwkv.infer.executor.sample", skip_all))]
fn sample_active<B: Backend + RapidSampleBackend>(
    logits: Tensor<B, 2>,
    state: &mut BatchState<B>,
    active_indices: &[usize],
    samplings: &[SamplingConfig],
    vocab_size: usize,
    device: &B::Device,
) -> Result<Vec<(usize, i32)>> {
    let active_count = active_indices.len();

    let (rng_active, inv_temp_tensor, top_k_tensor, top_p_tensor, penalties) = {
        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.sample.build_params");

        // Gather RNG states for active positions.
        let rng_slices: Vec<Tensor<B, 1, Int>> = active_indices
            .iter()
            .map(|&bi| state.rng_states.clone().slice([bi..bi + 1]))
            .collect();
        let rng_active = Tensor::cat(rng_slices, 0);

        // Build per-batch normalized sampling parameter tensors on host, then upload.
        let mut inv_temps = Vec::with_capacity(active_count);
        let mut norm_top_ks = Vec::with_capacity(active_count);
        let mut norm_top_ps = Vec::with_capacity(active_count);
        let mut any_penalties = false;

        for s in samplings {
            let temp = s.temperature.clamp(MIN_TEMP, MAX_TEMP);
            inv_temps.push(1.0f32 / temp);
            let (tk, tp) = normalize_topk_topp(vocab_size, s.top_k, s.top_p);
            norm_top_ks.push(tk as i32);
            norm_top_ps.push(tp);
            if s.penalties_enabled() {
                any_penalties = true;
            }
        }

        let inv_temp_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(inv_temps, [active_count]),
            device,
        );
        let top_k_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(norm_top_ks, [active_count]),
            device,
        );
        let top_p_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(norm_top_ps, [active_count]),
            device,
        );

        // Build penalties: use penalty kernel when ANY entry has penalties enabled.
        let penalties = if any_penalties {
            let pen_slices: Vec<Tensor<B, 2>> = active_indices
                .iter()
                .map(|&bi| state.penalties.clone().slice([bi..bi + 1, 0..vocab_size]))
                .collect();
            let penalties_tensor = Tensor::cat(pen_slices, 0);

            let mut presence_vals = Vec::with_capacity(active_count);
            let mut repetition_vals = Vec::with_capacity(active_count);
            let mut decay_vals = Vec::with_capacity(active_count);
            for s in samplings {
                presence_vals.push(s.presence_penalty);
                repetition_vals.push(s.repetition_penalty);
                decay_vals.push(s.penalty_decay);
            }

            Some((
                penalties_tensor,
                Tensor::<B, 1>::from_data(TensorData::new(presence_vals, [active_count]), device),
                Tensor::<B, 1>::from_data(TensorData::new(repetition_vals, [active_count]), device),
                Tensor::<B, 1>::from_data(TensorData::new(decay_vals, [active_count]), device),
            ))
        } else {
            None
        };

        (rng_active, inv_temp_tensor, top_k_tensor, top_p_tensor, penalties)
    };

    #[cfg(feature = "nsys")]
    let _nvtx_sample = nvtx::range!("rwkv.infer.executor.sample");
    let out = rapid_sample::<B>(
        logits,
        rng_active,
        inv_temp_tensor,
        top_k_tensor,
        top_p_tensor,
        penalties,
    );

    // Scatter RNG states back.
    {
        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.sample.scatter");
        for (i, &bi) in active_indices.iter().enumerate() {
            let s = out.states.clone().slice([i..i + 1]);
            state.rng_states = state
                .rng_states
                .clone()
                .slice_assign([bi..bi + 1], s);
        }

        // Scatter penalties back.
        if let Some(ref updated_penalties) = out.penalties {
            for (i, &bi) in active_indices.iter().enumerate() {
                let row = updated_penalties.clone().slice([i..i + 1, 0..vocab_size]);
                state.penalties = state
                    .penalties
                    .clone()
                    .slice_assign([bi..bi + 1, 0..vocab_size], row);
            }
        }
    }

    let token_ids = out
        .token_ids
        .to_data()
        .to_vec::<i32>()
        .expect("token_ids to_vec");

    let mut pairs = Vec::with_capacity(active_count);
    for (i, &bi) in active_indices.iter().enumerate() {
        pairs.push((bi, token_ids[i]));
    }
    Ok(pairs)
}

/// Reset a single batch position's state to zeros.
fn reset_position<B: Backend>(
    state: &mut BatchState<B>,
    batch_index: usize,
    max_batch_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
    vocab_size: usize,
    device: &B::Device,
) -> Result<()> {
    if batch_index >= max_batch_size {
        return Err(Error::BadRequest(format!(
            "reset: batch_index {batch_index} out of range (max_batch_size={max_batch_size})",
        )));
    }

    let zeros_shift = Tensor::<B, 2>::zeros([1, embedded_dim], device);
    for t in state.embedded_token_shift_for_time_mix.iter_mut() {
        *t = t
            .clone()
            .slice_assign([batch_index..batch_index + 1], zeros_shift.clone());
    }
    for t in state.embedded_token_shift_for_channel_mix.iter_mut() {
        *t = t
            .clone()
            .slice_assign([batch_index..batch_index + 1], zeros_shift.clone());
    }

    let zeros_state =
        Tensor::<B, 4>::zeros([1, num_heads, head_size, head_size], device);
    for s in state.wkv_state.iter_mut() {
        *s = s.clone().slice_assign([batch_index..batch_index + 1], zeros_state.clone());
    }

    let seed = (batch_index as i32).wrapping_add(1);
    let seed_tensor =
        Tensor::<B, 1, Int>::from_data(TensorData::new(vec![seed], [1]), device);
    state.rng_states = state
        .rng_states
        .clone()
        .slice_assign([batch_index..batch_index + 1], seed_tensor);

    let zeros_pen = Tensor::<B, 2>::zeros([1, vocab_size], (device, DType::F32));
    state.penalties = state
        .penalties
        .clone()
        .slice_assign([batch_index..batch_index + 1, 0..vocab_size], zeros_pen);

    Ok(())
}

// ---------------------------------------------------------------------------
// ModelForward adapter
// ---------------------------------------------------------------------------

pub struct RwkvLmModelForward<B: Backend> {
    model: AutoRegressiveModel<B>,
    state: BatchState<B>,
    device: B::Device,
    max_batch_size: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B> RwkvLmModelForward<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    pub fn new(
        device: B::Device,
        model: AutoRegressiveModel<B>,
        model_cfg: Arc<FinalModelConfig>,
        max_batch_size: usize,
    ) -> Self {
        let num_cells = model_cfg.num_cells;
        let vocab_size = model_cfg.vocab_size;
        let embedded_dim = model_cfg.embedded_dim;
        let num_heads = model_cfg.num_heads;
        let head_size = model_cfg.head_size_auto;

        let embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
            .collect();
        let wkv_state: Vec<Tensor<B, 4>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, num_heads, head_size, head_size], &device))
            .collect();
        let embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>> = (0..num_cells)
            .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
            .collect();

        let rng_init: Vec<i32> = (0..max_batch_size)
            .map(|i| (i as i32).wrapping_add(1))
            .collect();
        let rng_states =
            Tensor::<B, 1, Int>::from_data(TensorData::new(rng_init, [max_batch_size]), &device);

        let penalties = Tensor::<B, 2>::zeros([max_batch_size, vocab_size], (&device, DType::F32));

        let state = BatchState {
            embedded_token_shift_for_time_mix,
            wkv_state,
            embedded_token_shift_for_channel_mix,
            rng_states,
            penalties,
        };

        Self {
            model,
            state,
            device,
            max_batch_size,
            vocab_size,
            embedded_dim,
            num_heads,
            head_size,
        }
    }
}

impl<B> ModelForward for RwkvLmModelForward<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    fn forward(
        &mut self,
        batch_positions: &[(usize, &[i32], &[u8])],
        samplings: &[SamplingConfig],
        need_sample: bool,
    ) -> Result<Vec<(usize, i32)>> {
        rwkv_bench::trace_scope!("rwkv.infer.executor.forward_step");

        for (bi, _, _) in batch_positions {
            if *bi >= self.max_batch_size {
                return Err(Error::BadRequest(format!(
                    "forward: batch_index {bi} out of range (max_batch_size={})",
                    self.max_batch_size
                )));
            }
        }

        let mode = if need_sample {
            UnembedMode::LastToken
        } else {
            UnembedMode::Skip
        };

        let logits = forward_active(
            &self.model,
            &mut self.state,
            batch_positions,
            mode,
            &self.device,
        )?;

        if let Some(logits) = logits {
            let indices: Vec<usize> = batch_positions.iter().map(|(i, _, _)| *i).collect();
            sample_active(logits, &mut self.state, &indices, samplings, self.vocab_size, &self.device)
        } else {
            Ok(Vec::new())
        }
    }

    fn reset(&mut self, batch_index: usize) -> Result<()> {
        reset_position(
            &mut self.state,
            batch_index,
            self.max_batch_size,
            self.embedded_dim,
            self.num_heads,
            self.head_size,
            self.vocab_size,
            &self.device,
        )
    }
}

// ---------------------------------------------------------------------------
// ModelEngineFactory — builds engine groups for RuntimeManager
// ---------------------------------------------------------------------------

fn required_field<T: Copy>(
    value: Option<T>,
    field: &str,
    model_name: &str,
) -> Result<T> {
    value.ok_or_else(|| {
        Error::BadRequest(format!(
            "missing {} for model {} in infer config",
            field, model_name
        ))
    })
}

pub struct RwkvLmEngineFactory<B> {
    _marker: PhantomData<B>,
}

impl<B> RwkvLmEngineFactory<B> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B> ModelEngineFactory for RwkvLmEngineFactory<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend + Send + Sync,
{
    fn build_model_groups(
        &self,
        models: &[GenerationConfig],
        model_cfgs: &HashMap<String, Arc<FinalModelConfig>>,
    ) -> Result<HashMap<String, ModelRuntimeGroup>> {
        build_model_group_engines(models, |generation_cfg| {
            let model_cfg = model_cfgs.get(&generation_cfg.model_name).ok_or_else(|| {
                Error::Internal(format!(
                    "missing model cfg for {}",
                    generation_cfg.model_name
                ))
            })?;

            let device_type = required_field(
                generation_cfg.device_type,
                "device_type",
                &generation_cfg.model_name,
            )?;
            let max_batch_size = required_field(
                generation_cfg.max_batch_size,
                "max_batch_size",
                &generation_cfg.model_name,
            )?;
            let paragraph_len = required_field(
                generation_cfg.paragraph_len,
                "paragraph_len",
                &generation_cfg.model_name,
            )?;
            let max_context_len = required_field(
                generation_cfg.max_context_len,
                "max_context_len",
                &generation_cfg.model_name,
            )?;
            let decode_first = required_field(
                generation_cfg.decode_first,
                "decode_first",
                &generation_cfg.model_name,
            )?;

            let mut engines = Vec::new();
            for device_id in &generation_cfg.device_ids {
                let device = B::Device::from_id(DeviceId::new(device_type, *device_id));

                let model_config = AutoRegressiveModelConfig::new(
                    model_cfg.num_cells,
                    model_cfg.vocab_size,
                    model_cfg.embedded_dim,
                    model_cfg.num_heads,
                    model_cfg.head_size_auto,
                );
                let model_runtime = model_config.init::<B>(&device);
                let model_runtime = model_runtime
                    .load_file(
                        &generation_cfg.weights_path,
                        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                        &device,
                    )
                    .map_err(|e| {
                        Error::Internal(format!(
                            "failed to load weights {} for model {}: {e}",
                            generation_cfg.weights_path, generation_cfg.model_name
                        ))
                    })?;

                let executor = RwkvLmModelForward::<B>::new(
                    device.clone(),
                    model_runtime,
                    model_cfg.clone(),
                    max_batch_size,
                );

                let engine = EngineRuntime::spawn(
                    EngineRuntimeConfig {
                        tokenizer_vocab_path: generation_cfg.tokenizer_vocab_path.clone(),
                        max_batch_size,
                        paragraph_len,
                        max_context_len,
                        decode_first,
                    },
                    Box::new(executor),
                )?;

                log::info!(
                    "engine ready: model_name={} model_cfg={} device_type={} device_id={} \
                     weights_path={}",
                    generation_cfg.model_name,
                    generation_cfg.model_cfg,
                    device_type,
                    device_id,
                    generation_cfg.weights_path
                );
                engines.push(Arc::new(engine));
            }

            Ok(engines)
        })
    }
}
