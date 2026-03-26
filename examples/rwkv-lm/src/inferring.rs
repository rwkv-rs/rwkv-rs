use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use itertools::izip;
use serde::de::DeserializeOwned;
use rwkv::{
    config::{
        default_cfg_dir,
        get_arg_value,
        raw::{
            infer::{GenerationConfig, RawInferConfig},
            model::RawModelConfig,
        },
        validated::model::{FinalModelConfig, FinalModelConfigBuilder},
    },
    custom::{
        Tensor,
        cubecl::device::DeviceId,
        prelude::{Backend, DeviceOps, Int, TensorData},
        store::{BurnpackStore, ModuleSnapshot},
        tensor::{DType, IndexingUpdateOp},
    },
    data::tokenizer::Tokenizer,
    infer::{
        cores::{
            forward::{
                ModelForward,
                StepMode,
                TokenId,
                logprobs::build_sampled_token_logprob,
                sampling::{SamplingConfigsTensor, sampling_configs_to_tensor},
            },
            queue::queue_worker::spawn_queue_worker,
        },
        handlers::auth::AuthConfig,
        routes::http_api::AppState,
        services::{QueueMap, QueueMapBuilder, ServiceError, ServiceResult},
    },
    nn::kernels::{
        addcmul::AddcmulBackend,
        rapid_sample::{RapidSampleBackend, rapid_sample},
        token_shift_diff::TokenShiftDiffBackend,
        wkv7_common::Wkv7Backend,
    },
};
#[cfg(feature = "ipc")]
use rwkv::infer::routes::IpcServerConfig;

use crate::model::{AutoRegressiveModel, AutoRegressiveModelConfig, UnembedMode};

rwkv::custom_mode!();

const GUIDED_MASKED_LOGIT: f32 = -1.0e30;

pub fn infer_cli_args(args: &[String]) -> (PathBuf, String) {
    let config_dir = get_arg_value(args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_cfg_dir);
    let infer_cfg =
        get_arg_value(args, "--infer-cfg").unwrap_or_else(|| "rwkv-7.2b-g1e".to_string());
    (config_dir, infer_cfg)
}

pub struct RwkvLmForward<B: Backend> {
    model: AutoRegressiveModel<B>,
    embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>>,
    state: Vec<Tensor<B, 4>>,
    embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>>,
    rng: Tensor<B, 1, Int>,
    penalties: Tensor<B, 2>,
    device: B::Device,

    max_batch_size: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend + Wkv7Backend + RapidSampleBackend> RwkvLmForward<B> {
    pub fn new(
        model: AutoRegressiveModel<B>,
        model_cfg: Arc<FinalModelConfig>,
        max_batch_size: usize,
        device: B::Device,
    ) -> Self {
        let num_cells = model_cfg.num_cells;
        let vocab_size = model_cfg.vocab_size;
        let embedded_dim = model_cfg.embedded_dim;
        let num_heads = model_cfg.num_heads;
        let head_size = model_cfg.head_size_auto;

        let init_embedded_token_shift = || {
            (0..num_cells)
                .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
                .collect::<Vec<_>>()
        };
        let init_state = || {
            (0..num_cells)
                .map(|_| Tensor::zeros([max_batch_size, num_heads, head_size, head_size], &device))
                .collect::<Vec<_>>()
        };

        let rng = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                (0..max_batch_size)
                    .map(|i| (i as i32).wrapping_add(1))
                    .collect(),
                [max_batch_size],
            ),
            &device,
        );
        let penalties = Tensor::<B, 2>::zeros([max_batch_size, vocab_size], (&device, DType::F32));

        Self {
            model,
            embedded_token_shift_for_time_mix: init_embedded_token_shift(),
            state: init_state(),
            embedded_token_shift_for_channel_mix: init_embedded_token_shift(),
            rng,
            penalties,
            device,
            max_batch_size,
            vocab_size,
            embedded_dim,
            num_heads,
            head_size,
        }
    }
}

impl<B> ModelForward for RwkvLmForward<B>
where
    B: Backend + TokenShiftDiffBackend + AddcmulBackend + Wkv7Backend + RapidSampleBackend,
{
    fn step(
        &mut self,
        batch_ids: &[usize],
        contexts: &[&[i32]],
        context_masks: &[&[u8]],
        mode: StepMode<'_>,
    ) -> Option<Vec<TokenId>> {
        assert_eq!(batch_ids.len(), contexts.len());
        assert_eq!(contexts.len(), context_masks.len());
        if batch_ids.is_empty() {
            return match mode {
                StepMode::PrefillNoOutput => None,
                StepMode::Sample { .. } => Some(Vec::new()),
            };
        }
        rwkv_bench::trace_scope!("rwkv.infer.executor.step_infer");
        assert!(
            batch_ids.iter().all(|&id| id < self.max_batch_size),
            "batch index out of range"
        );

        let context_len = contexts[0].len();
        assert!(context_len > 0, "empty context is not allowed");
        for (context, mask) in contexts.iter().zip(context_masks.iter()) {
            assert_eq!(context.len(), context_len);
            assert_eq!(mask.len(), context_len);
        }

        let batch_size = batch_ids.len();
        let batch_masks: Tensor<B, 1> = {
            let mut batch_masks = vec![1.0f32; self.max_batch_size];
            for &batch_id in batch_ids {
                batch_masks[batch_id] = 0.0;
            }
            Tensor::from_data(
                TensorData::new(batch_masks, [self.max_batch_size]),
                &self.device,
            )
        };
        let batch_ids_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(
                batch_ids.iter().map(|&id| id as i32).collect::<Vec<i32>>(),
                [batch_size],
            ),
            &self.device,
        );
        let contexts: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(contexts.concat(), [batch_size, context_len]),
            &self.device,
        );
        let context_masks: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(
                context_masks.concat().iter().map(|&m| m as f32).collect(),
                [batch_size, context_len],
            ),
            &self.device,
        );

        let (
            mut embedded_token_shift_for_time_mix,
            mut state,
            mut embedded_token_shift_for_channel_mix,
        ) = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.executor.gather_state");
            (
                self.embedded_token_shift_for_time_mix
                    .iter()
                    .map(|x| x.clone().select(0, batch_ids_tensor.clone()))
                    .collect(),
                self.state
                    .iter()
                    .map(|x| x.clone().select(0, batch_ids_tensor.clone()))
                    .collect(),
                self.embedded_token_shift_for_channel_mix
                    .iter()
                    .map(|x| x.clone().select(0, batch_ids_tensor.clone()))
                    .collect(),
            )
        };

        let logits = self.model.infer(
            contexts,
            Some(context_masks),
            &mut embedded_token_shift_for_time_mix,
            &mut state,
            &mut embedded_token_shift_for_channel_mix,
            UnembedMode::LastToken,
        );

        let batch_masks_2d = batch_masks.clone().unsqueeze_dim::<2>(1);
        let batch_masks_4d = batch_masks
            .clone()
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);

        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.scatter_state");
        for (
            cell_idx,
            (
                cell_embedded_token_shift_for_time_mix,
                cell_state,
                cell_embedded_token_shift_for_channel_mix,
            ),
        ) in izip!(
            embedded_token_shift_for_time_mix,
            state,
            embedded_token_shift_for_channel_mix,
        )
        .enumerate()
        {
            self.embedded_token_shift_for_time_mix[cell_idx] =
                (self.embedded_token_shift_for_time_mix[cell_idx].clone() * batch_masks_2d.clone())
                    .select_assign(
                        0,
                        batch_ids_tensor.clone(),
                        cell_embedded_token_shift_for_time_mix,
                        IndexingUpdateOp::Add,
                    );

            self.state[cell_idx] = (self.state[cell_idx].clone() * batch_masks_4d.clone())
                .select_assign(
                    0,
                    batch_ids_tensor.clone(),
                    cell_state,
                    IndexingUpdateOp::Add,
                );

            self.embedded_token_shift_for_channel_mix[cell_idx] =
                (self.embedded_token_shift_for_channel_mix[cell_idx].clone()
                    * batch_masks_2d.clone())
                .select_assign(
                    0,
                    batch_ids_tensor.clone(),
                    cell_embedded_token_shift_for_channel_mix,
                    IndexingUpdateOp::Add,
                );
        }

        let logits = logits.map(|logits| logits.squeeze_dim::<2>(1));

        match mode {
            StepMode::PrefillNoOutput => None,
            StepMode::Sample {
                sampling_configs,
                token_logprobs_configs,
                guided_token_masks,
            } => {
                let Some(logits) = logits else {
                    return Some(Vec::new());
                };
                rwkv_bench::trace_scope!("rwkv.infer.executor.sample");
                assert_eq!(batch_ids.len(), sampling_configs.len());
                assert_eq!(batch_ids.len(), token_logprobs_configs.len());
                assert_eq!(batch_ids.len(), guided_token_masks.len());

                let logits = if guided_token_masks.iter().all(Option::is_none) {
                    logits
                } else {
                    let mut logits_vec = logits.to_data().to_vec::<f32>().unwrap();
                    assert_eq!(logits_vec.len(), guided_token_masks.len() * self.vocab_size);

                    for (row_index, guided_token_mask) in guided_token_masks.iter().enumerate() {
                        let Some(guided_token_mask) = guided_token_mask else {
                            continue;
                        };

                        let row_start = row_index * self.vocab_size;
                        let row_end = row_start + self.vocab_size;
                        let row = &mut logits_vec[row_start..row_end];

                        for (token_id, logit) in row.iter_mut().enumerate() {
                            let word_index = token_id / 32;
                            let bit_index = token_id % 32;
                            let allowed = guided_token_mask
                                .get(word_index)
                                .is_some_and(|word| (word & (1 << bit_index)) != 0);
                            if !allowed {
                                *logit = GUIDED_MASKED_LOGIT;
                            }
                        }
                    }

                    Tensor::from_data(
                        TensorData::new(logits_vec, [guided_token_masks.len(), self.vocab_size]),
                        &self.device,
                    )
                };

                let rng = self.rng.clone().select(0, batch_ids_tensor.clone());
                let penalties = self.penalties.clone().select(0, batch_ids_tensor.clone());
                let SamplingConfigsTensor {
                    inv_temperatures,
                    top_ks,
                    top_ps,
                    presence_penalties,
                    repetition_penalties,
                    penalties_decay,
                } = sampling_configs_to_tensor::<B>(
                    sampling_configs,
                    self.vocab_size,
                    &self.device,
                );

                let sample_output = rapid_sample::<B>(
                    logits,
                    rng,
                    inv_temperatures,
                    top_ks,
                    top_ps,
                    Some((
                        penalties,
                        presence_penalties,
                        repetition_penalties,
                        penalties_decay,
                    )),
                );

                self.rng = (self.rng.clone().float() * batch_masks.clone())
                    .select_assign(
                        0,
                        batch_ids_tensor.clone(),
                        sample_output.states.clone().cast(DType::I32).float(),
                        IndexingUpdateOp::Add,
                    )
                    .int();
                if let Some(ref updated_penalties) = sample_output.penalties {
                    let penalties_mask_2d = batch_masks_2d.clone().cast(DType::F32);
                    self.penalties = (self.penalties.clone() * penalties_mask_2d).select_assign(
                        0,
                        batch_ids_tensor.clone(),
                        updated_penalties.clone(),
                        IndexingUpdateOp::Add,
                    );
                }

                let token_ids = sample_output.token_ids.to_data().to_vec::<i32>().unwrap();
                let mut logprobs = vec![None; batch_ids.len()];
                let requested_rows: Vec<usize> = token_logprobs_configs
                    .iter()
                    .enumerate()
                    .filter_map(|(row_index, cfg)| cfg.as_ref().map(|_| row_index))
                    .collect();

                if !requested_rows.is_empty() {
                    let requested_rows_tensor = Tensor::<B, 1, Int>::from_data(
                        TensorData::new(
                            requested_rows
                                .iter()
                                .map(|&row_index| row_index as i32)
                                .collect::<Vec<i32>>(),
                            [requested_rows.len()],
                        ),
                        &self.device,
                    );
                    let requested_probs = sample_output
                        .probs
                        .select(0, requested_rows_tensor)
                        .to_data()
                        .to_vec::<f32>()
                        .unwrap();

                    for (selected_row_index, row_index) in requested_rows.into_iter().enumerate() {
                        let row_start = selected_row_index * self.vocab_size;
                        let row_end = row_start + self.vocab_size;
                        logprobs[row_index] = Some(build_sampled_token_logprob(
                            &requested_probs[row_start..row_end],
                            token_ids[row_index],
                            token_logprobs_configs[row_index].as_ref().unwrap(),
                        ));
                    }
                }

                Some(
                    batch_ids
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(row_index, batch_index)| TokenId {
                            batch_index,
                            token_id: token_ids[row_index],
                            logprob: logprobs[row_index].clone(),
                            finish_after_token: false,
                        })
                        .collect(),
                )
            }
        }
    }

    fn reset(&mut self, batch_index: usize) {
        assert!(
            batch_index < self.max_batch_size,
            "batch index out of range"
        );

        let zeros_shift = Tensor::<B, 2>::zeros([1, self.embedded_dim], &self.device);
        for buffer in &mut self.embedded_token_shift_for_time_mix {
            *buffer = buffer
                .clone()
                .slice_assign([batch_index..batch_index + 1], zeros_shift.clone());
        }
        for buffer in &mut self.embedded_token_shift_for_channel_mix {
            *buffer = buffer
                .clone()
                .slice_assign([batch_index..batch_index + 1], zeros_shift.clone());
        }

        let zeros_state = Tensor::<B, 4>::zeros(
            [1, self.num_heads, self.head_size, self.head_size],
            &self.device,
        );
        for state in &mut self.state {
            *state = state
                .clone()
                .slice_assign([batch_index..batch_index + 1], zeros_state.clone());
        }

        let seed = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![(batch_index as i32).wrapping_add(1)], [1]),
            &self.device,
        );
        self.rng = self
            .rng
            .clone()
            .slice_assign([batch_index..batch_index + 1], seed);

        let zeros_pen = Tensor::<B, 2>::zeros([1, self.vocab_size], (&self.device, DType::F32));
        self.penalties = self.penalties.clone().slice_assign(
            [batch_index..batch_index + 1, 0..self.vocab_size],
            zeros_pen,
        );
    }
}

pub fn build_http_runtime<B>(config_dir: PathBuf, infer_cfg_name: &str) -> (AppState, String)
where
    B: Backend
        + TokenShiftDiffBackend
        + AddcmulBackend
        + Wkv7Backend
        + RapidSampleBackend
        + Send
        + Sync
        + 'static,
{
    let infer_cfg_path = config_dir
        .join("infer")
        .join(format!("{infer_cfg_name}.toml"));
    let infer_cfg = load_raw_infer_cfg(&infer_cfg_path).unwrap_or_else(|err| {
        panic!(
            "failed to load infer config {}: {}",
            infer_cfg_path.display(),
            err.body().error.message
        )
    });
    let queues =
        build_queue_map_from_generation_cfgs::<B>(&config_dir, &infer_cfg_path, &infer_cfg.models)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to build infer queues from {}: {}",
                    infer_cfg_path.display(),
                    err.body().error.message
                )
            });
    let build_queues: QueueMapBuilder = Arc::new({
        let config_dir = config_dir.clone();
        let infer_cfg_path = infer_cfg_path.clone();
        move |models| {
            build_queue_map_from_generation_cfgs::<B>(&config_dir, &infer_cfg_path, models)
        }
    });
    let bind_addr = infer_cfg.http_bind_addr.clone().unwrap();

    (
        AppState::new(
            AuthConfig {
                api_key: infer_cfg.api_key.clone(),
            },
            infer_cfg.request_body_limit_bytes.unwrap(),
            infer_cfg.sse_keep_alive_ms.unwrap(),
            infer_cfg.allowed_origins.clone(),
            queues,
        )
        .with_reload_support(infer_cfg_path, build_queues),
        bind_addr,
    )
}

#[cfg(feature = "ipc")]
pub fn build_ipc_server_config(
    config_dir: PathBuf,
    infer_cfg_name: &str,
) -> Option<IpcServerConfig> {
    let infer_cfg_path = config_dir
        .join("infer")
        .join(format!("{infer_cfg_name}.toml"));
    let mut infer_cfg: RawInferConfig = read_toml_file(&infer_cfg_path).ok()?;
    infer_cfg.fill_default();

    let ipc_cfg = infer_cfg.ipc?;
    if !ipc_cfg.enabled.unwrap_or(false) {
        return None;
    }

    Some(IpcServerConfig {
        service_name: ipc_cfg.service_name.unwrap(),
        max_request_bytes: ipc_cfg.max_request_bytes.unwrap(),
        max_response_bytes: ipc_cfg.max_response_bytes.unwrap(),
        max_inflight_requests: ipc_cfg.max_inflight_requests.unwrap(),
        require_api_key: ipc_cfg.require_api_key.unwrap(),
    })
}

fn build_queue_map_from_generation_cfgs<B>(
    config_dir: &Path,
    infer_cfg_path: &Path,
    models: &[GenerationConfig],
) -> ServiceResult<QueueMap>
where
    B: Backend
        + TokenShiftDiffBackend
        + AddcmulBackend
        + Wkv7Backend
        + RapidSampleBackend
        + Send
        + Sync
        + 'static,
{
    validate_generation_models(models)?;
    let infer_cfg_dir = infer_cfg_path.parent().unwrap_or_else(|| Path::new("."));
    let runtime_models = resolve_runtime_models(infer_cfg_dir, models);
    let model_cfgs = load_model_cfgs(config_dir, infer_cfg_dir, &runtime_models)?;
    Ok(build_queue_map::<B>(&runtime_models, &model_cfgs))
}

fn build_queue_map<B>(
    models: &[GenerationConfig],
    model_cfgs: &HashMap<String, Arc<FinalModelConfig>>,
) -> QueueMap
where
    B: Backend
        + TokenShiftDiffBackend
        + AddcmulBackend
        + Wkv7Backend
        + RapidSampleBackend
        + Send
        + Sync
        + 'static,
{
    let mut queue_map: QueueMap = HashMap::new();

    for generation_cfg in models {
        let model_cfg = model_cfgs.get(&generation_cfg.model_name).unwrap();
        let device_type = generation_cfg.device_type.unwrap();
        let max_batch_size = generation_cfg.max_batch_size.unwrap();
        let paragraph_len = generation_cfg.paragraph_len.unwrap();
        let tokenizer = Arc::new(Tokenizer::new(&generation_cfg.tokenizer_vocab_path).unwrap());

        for &device_id in &generation_cfg.device_ids {
            let device = B::Device::from_id(DeviceId::new(device_type, device_id));
            let model_config = AutoRegressiveModelConfig::new(
                model_cfg.num_cells,
                model_cfg.vocab_size,
                model_cfg.embedded_dim,
                model_cfg.num_heads,
                model_cfg.head_size_auto,
            );
            let mut model_runtime = model_config.init::<B>(&device);
            let mut store = BurnpackStore::from_file(&generation_cfg.weights_path).zero_copy(true);
            model_runtime.load_from(&mut store).unwrap();
            let executor = RwkvLmForward::<B>::new(
                model_runtime,
                model_cfg.clone(),
                max_batch_size,
                device.clone(),
            );
            let handle = spawn_queue_worker(
                Box::new(executor),
                Arc::clone(&tokenizer),
                max_batch_size,
                paragraph_len,
                device_id,
                generation_cfg.weights_path.clone(),
            );

            log::info!(
                "queue ready: model_name={} model_cfg={} device_type={} device_id={} weights_path={}",
                generation_cfg.model_name,
                generation_cfg.model_cfg,
                device_type,
                device_id,
                generation_cfg.weights_path
            );
            queue_map
                .entry(generation_cfg.model_name.clone())
                .or_default()
                .push(handle);
        }
    }

    queue_map
}

fn load_raw_infer_cfg(path: &Path) -> ServiceResult<RawInferConfig> {
    let mut cfg: RawInferConfig = read_toml_file(path)?;
    cfg.fill_default();
    validate_generation_models(&cfg.models)?;
    Ok(cfg)
}

fn read_toml_file<T: DeserializeOwned>(path: &Path) -> ServiceResult<T> {
    let content = fs::read_to_string(path).map_err(|err| {
        ServiceError::bad_request(format!("failed to read {}: {err}", path.display()))
    })?;
    toml::from_str(&content)
        .map_err(|err| ServiceError::bad_request(format!("invalid toml {}: {err}", path.display())))
}

fn validate_generation_models(models: &[GenerationConfig]) -> ServiceResult<()> {
    if models.is_empty() {
        return Err(ServiceError::bad_request(
            "infer config requires at least one model",
        ));
    }

    let mut names = HashSet::new();
    for model in models {
        if model.model_name.trim().is_empty() {
            return Err(ServiceError::bad_request("model_name cannot be empty"));
        }
        if !names.insert(model.model_name.clone()) {
            return Err(ServiceError::bad_request(format!(
                "duplicated model_name: {}",
                model.model_name
            )));
        }
        if model.model_cfg.trim().is_empty() {
            return Err(ServiceError::bad_request(format!(
                "model_cfg cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.weights_path.trim().is_empty() {
            return Err(ServiceError::bad_request(format!(
                "weights_path cannot be empty for model {}",
                model.model_name
            )));
        }
        match Path::new(&model.weights_path)
            .extension()
            .and_then(|extension| extension.to_str())
        {
            Some("bpk") => {}
            Some("mpk") => {
                return Err(ServiceError::bad_request(format!(
                    "weights_path for model {} points to unsupported .mpk file {}; convert it to .bpk first",
                    model.model_name, model.weights_path
                )));
            }
            _ => {
                return Err(ServiceError::bad_request(format!(
                    "weights_path for model {} must point to a .bpk file, got {}",
                    model.model_name, model.weights_path
                )));
            }
        }
        if model.tokenizer_vocab_path.trim().is_empty() {
            return Err(ServiceError::bad_request(format!(
                "tokenizer_vocab_path cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.device_ids.is_empty() {
            return Err(ServiceError::bad_request(format!(
                "device_ids cannot be empty for model {}",
                model.model_name
            )));
        }
        if model.max_batch_size.unwrap_or_default() < 1 {
            return Err(ServiceError::bad_request(format!(
                "max_batch_size must be >= 1 for model {}",
                model.model_name
            )));
        }
        if model.max_context_len.unwrap_or_default() < 1 {
            return Err(ServiceError::bad_request(format!(
                "max_context_len must be >= 1 for model {}",
                model.model_name
            )));
        }
    }

    Ok(())
}

fn load_model_cfgs(
    config_dir: &Path,
    infer_cfg_dir: &Path,
    models: &[GenerationConfig],
) -> ServiceResult<HashMap<String, Arc<FinalModelConfig>>> {
    let mut model_cfgs = HashMap::new();

    for model in models {
        let model_cfg_path = resolve_model_cfg_path(config_dir, infer_cfg_dir, &model.model_cfg);
        let mut raw_model_cfg: RawModelConfig = read_toml_file(&model_cfg_path)?;
        raw_model_cfg.fill_default();

        let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
        model_cfg_builder.fill_auto_after_load();
        model_cfgs.insert(model.model_name.clone(), model_cfg_builder.build_local());
    }

    Ok(model_cfgs)
}

fn resolve_runtime_models(
    infer_cfg_dir: &Path,
    models: &[GenerationConfig],
) -> Vec<GenerationConfig> {
    let mut resolved_models = models.to_vec();
    for model in &mut resolved_models {
        model.weights_path = resolve_path(infer_cfg_dir, &model.weights_path);
        model.tokenizer_vocab_path = resolve_path(infer_cfg_dir, &model.tokenizer_vocab_path);
    }
    resolved_models
}

fn resolve_model_cfg_path(config_dir: &Path, infer_cfg_dir: &Path, model_cfg: &str) -> PathBuf {
    if model_cfg.contains('/') || model_cfg.contains('\\') {
        let path = PathBuf::from(model_cfg);
        if path.is_absolute() {
            path
        } else {
            infer_cfg_dir.join(path)
        }
    } else {
        config_dir.join("model").join(format!("{model_cfg}.toml"))
    }
}

fn resolve_path(base_dir: &Path, path: &str) -> String {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        path.to_string()
    } else {
        base_dir.join(candidate).to_string_lossy().to_string()
    }
}
