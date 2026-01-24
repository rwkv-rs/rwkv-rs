use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
#[cfg(feature = "backend_cuda")]
use burn::backend::cuda::Cuda;
#[cfg(feature = "backend_ndarray")]
use burn::backend::ndarray::{FloatNdArrayElement, IntNdArrayElement, NdArray, QuantElement};
#[cfg(any(feature = "backend_wgpu", feature = "backend_metal"))]
use burn::backend::wgpu::Wgpu;
#[cfg(any(feature = "backend_cuda", feature = "backend_wgpu", feature = "backend_metal"))]
use burn::cubecl::{FloatElement, IntElement};
#[cfg(any(feature = "backend_wgpu", feature = "backend_metal"))]
use burn::cubecl::BoolElement;
use burn::prelude::Backend;
use burn_train::{
    Interrupter,
    logger::{AsyncLogger, FileLogger, Logger},
    renderer::{MetricsRenderer, tui::TuiMetricsRenderer},
};
use chrono::Local;
use log::{info, warn};
use rwkv_config::{
    load_toml,
    raw::{model::RawModelConfig, train::RawTrainConfig},
    validated::{
        model::FinalModelConfigBuilder,
        train::{FinalTrainConfigBuilder, TRAIN_CFG},
    },
};
use tokio::runtime::Runtime;
use wandb::LogData;

use crate::{
    logger::wandb::{WandbLoggerConfig, init_logger},
    renderer::BarMetricsRenderer,
    utils::{auto_create_directory, read_record_file},
};

pub fn init_cfg<P1: AsRef<Path>, P2: AsRef<Path>>(
    model_cfg_path: P1,
    train_cfg_path: P2,
) -> (FinalModelConfigBuilder, FinalTrainConfigBuilder) {
    let mut raw_model_cfg: RawModelConfig = load_toml(model_cfg_path);
    let mut raw_train_cfg: RawTrainConfig = load_toml(train_cfg_path);

    raw_model_cfg.fill_default();
    raw_train_cfg.fill_default();

    let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
    let mut train_cfg_builder = FinalTrainConfigBuilder::load_from_raw(raw_train_cfg);

    model_cfg_builder.fill_auto_after_load();
    train_cfg_builder.fill_auto_after_load();

    (model_cfg_builder, train_cfg_builder)
}

pub fn init_log(train_cfg_builder: &mut FinalTrainConfigBuilder) -> PathBuf {
    let full_experiment_log_path = auto_create_directory(
        auto_create_directory(PathBuf::from(
            train_cfg_builder.get_experiment_log_base_path().unwrap(),
        ))
        .join(train_cfg_builder.get_experiment_name().unwrap()),
    )
    .canonicalize()
    .unwrap();

    let level = train_cfg_builder.get_level().unwrap();
    let _guard = clia_tracing_config::build()
        .filter_level(level.as_str())
        .with_ansi(true)
        .to_stdout(true)
        .directory(full_experiment_log_path.to_str().unwrap())
        .file_name("experiment.log")
        .init();

    info!("log level: {}", level);

    let record_path = read_record_file(
        train_cfg_builder.get_record_path(),
        &full_experiment_log_path,
    );
    info!("Getting Record Path Completed. record_path: {record_path:?}");
    train_cfg_builder.fill_after_read_record_file(record_path);

    full_experiment_log_path
}


pub trait BackendDeviceInit: Backend {
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device>;
}

impl<B, C> BackendDeviceInit for Autodiff<B, C>
where
    B: BackendDeviceInit,
    C: CheckpointStrategy,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        B::init_devices(train_cfg_builder)
    }
}

#[cfg(feature = "backend_ndarray")]
impl<E, I, Q> BackendDeviceInit for NdArray<E, I, Q>
where
    E: FloatNdArrayElement,
    I: IntNdArrayElement,
    Q: QuantElement,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        let seed = train_cfg_builder.get_random_seed().unwrap();
        (0..train_cfg_builder.get_num_devices_per_node().unwrap())
            .map(|_| {
                let device = burn::backend::ndarray::NdArrayDevice::Cpu;
                Self::seed(&device, seed);
                device
            })
            .collect()
    }
}

#[cfg(feature = "backend_cuda")]
impl<F, I> BackendDeviceInit for Cuda<F, I>
where
    F: FloatElement,
    I: IntElement,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        let seed = train_cfg_builder.get_random_seed().unwrap();
        (0..train_cfg_builder.get_num_devices_per_node().unwrap())
            .map(|i| {
                let device = burn::backend::cuda::CudaDevice::new(i);
                Self::seed(&device, seed);
                device
            })
            .collect()
    }
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_metal"))]
impl<F, I, B> BackendDeviceInit for Wgpu<F, I, B>
where
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        let seed = train_cfg_builder.get_random_seed().unwrap();
        let requested = train_cfg_builder.get_num_devices_per_node().unwrap();
        let devices = if requested <= 1 {
            vec![burn::backend::wgpu::WgpuDevice::default()]
        } else {
            (0..requested)
                .map(|i| burn::backend::wgpu::WgpuDevice::DiscreteGpu(i))
                .collect::<Vec<_>>()
        };

        for device in &devices {
            #[cfg(feature = "backend_metal")]
            burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                device,
                Default::default(),
            );
            #[cfg(all(not(feature = "backend_metal"), feature = "backend_wgpu"))]
            burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                device,
                Default::default(),
            );

            Self::seed(device, seed);
        }

        devices
    }
}

pub fn init_devices<B: BackendDeviceInit>(
    train_cfg_builder: &FinalTrainConfigBuilder,
) -> Vec<B::Device> {
    B::init_devices(train_cfg_builder)
}

pub fn init_file_logger(exp_log_path: &Path) -> AsyncLogger<String> {
    let mut file_logger =
        AsyncLogger::new(FileLogger::new(exp_log_path.join("training_metrics.log")));

    file_logger.log("global_step,epoch,step_in_epoch,learning_rate,loss".to_string());
    file_logger
}

pub fn init_wandb_logger() -> Option<AsyncLogger<LogData>> {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let mut wandb_logger: Option<AsyncLogger<LogData>> = None;
    if TRAIN_CFG.get().unwrap().upload_to_wandb {
        let api_key = TRAIN_CFG.get().unwrap().wandb_api_key.as_ref().unwrap();
        let project = TRAIN_CFG
            .get()
            .unwrap()
            .wandb_project_name
            .as_ref()
            .unwrap();
        let entity = TRAIN_CFG.get().unwrap().wandb_entity_name.as_ref();

        let run_name = format!("{}_{}", TRAIN_CFG.get().unwrap().experiment_name, timestamp);
        let mut config = WandbLoggerConfig::new(api_key, project).run_name(run_name);
        if let Some(entity) = entity {
            config = config.entity(entity);
        } else {
            warn!("wandb entity name missing, falling back to default entity");
        }
        let rt = Runtime::new().unwrap();
        wandb_logger = rt.block_on(async { Some(init_logger(config).await) });
        info!("Wandb logger initialized.");
    }
    wandb_logger
}

pub fn init_renderer() -> (Interrupter, Box<dyn MetricsRenderer>) {
    let interrupter = Interrupter::new();
    if TRAIN_CFG.get().unwrap().use_tui {
        (
            interrupter.clone(),
            Box::new(TuiMetricsRenderer::new(interrupter, None)),
        )
    } else {
        (
            interrupter,
            Box::new(BarMetricsRenderer::new(
                TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
            )),
        )
    }
}
