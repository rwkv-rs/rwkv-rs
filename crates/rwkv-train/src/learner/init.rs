#[cfg(feature = "wgpu")]
use std::any::TypeId;
use std::{
    any::Any,
    path::{Path, PathBuf},
};

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
#[cfg(feature = "wgpu")]
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::Backend;
#[cfg(feature = "cubecl")]
use burn_cubecl::cubecl::device::{Device as CubeDevice, DeviceId};
#[cfg(feature = "cubecl")]
use burn_cubecl::{CubeBackend, CubeRuntime};
#[cfg(feature = "fusion")]
use burn_fusion::{Fusion, FusionBackend};
#[cfg(feature = "tui")]
use burn_train::renderer::tui::TuiMetricsRenderer;
use burn_train::{
    Interrupter,
    logger::{AsyncLogger, FileLogger, Logger},
    renderer::MetricsRenderer,
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
use wandb::LogData;

use crate::{
    logger::wandb::{
        WandbLogger, WandbLoggerConfig, init_logger_blocking, init_metric_logger_blocking,
    },
    renderer::BarMetricsRenderer,
    utils::{auto_create_directory, read_record_file},
};

fn is_path_like(s: &str) -> bool {
    s.contains('/') || s.contains('\\')
}

fn resolve_path(base_dir: &Path, path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        base_dir.join(p).to_string_lossy().to_string()
    }
}

fn resolve_model_cfg_path(config_dir: &Path, train_cfg_dir: &Path, model_cfg: &str) -> PathBuf {
    if is_path_like(model_cfg) {
        let p = PathBuf::from(model_cfg);
        if p.is_absolute() {
            p
        } else {
            train_cfg_dir.join(p)
        }
    } else {
        config_dir.join("model").join(format!("{model_cfg}.toml"))
    }
}

pub fn init_cfg<P: AsRef<Path>>(
    config_dir: P,
    train_cfg_name: &str,
) -> (FinalModelConfigBuilder, FinalTrainConfigBuilder) {
    let config_dir = config_dir.as_ref();
    let train_cfg_path = config_dir
        .join("train")
        .join(format!("{train_cfg_name}.toml"));
    let train_cfg_dir = train_cfg_path.parent().unwrap_or_else(|| Path::new("."));

    let mut raw_train_cfg: RawTrainConfig = load_toml(&train_cfg_path);
    raw_train_cfg.fill_default();

    // Resolve relative paths against the train config directory.
    raw_train_cfg.dataset_base_path = resolve_path(train_cfg_dir, &raw_train_cfg.dataset_base_path);
    raw_train_cfg.experiment_log_base_path = raw_train_cfg
        .experiment_log_base_path
        .map(|p| resolve_path(train_cfg_dir, &p));
    raw_train_cfg.record_path = raw_train_cfg
        .record_path
        .map(|p| resolve_path(train_cfg_dir, &p));

    let model_cfg_path =
        resolve_model_cfg_path(config_dir, train_cfg_dir, raw_train_cfg.model_cfg.as_str());

    let mut raw_model_cfg: RawModelConfig = load_toml(&model_cfg_path);
    raw_model_cfg.fill_default();

    let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
    let mut train_cfg_builder = FinalTrainConfigBuilder::load_from_raw(raw_train_cfg);

    model_cfg_builder.fill_auto_after_load();
    train_cfg_builder.fill_auto_after_load();

    (model_cfg_builder, train_cfg_builder)
}

pub fn init_cfg_paths<P1: AsRef<Path>, P2: AsRef<Path>>(
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
    #[cfg(feature = "trace")]
    let tracing_already_initialized = tracing::dispatcher::has_been_set();
    #[cfg(not(feature = "trace"))]
    let tracing_already_initialized = false;

    if !tracing_already_initialized {
        let _guard = clia_tracing_config::build()
            .filter_level(level.as_str())
            .with_ansi(true)
            .to_stdout(true)
            .directory(full_experiment_log_path.to_str().unwrap())
            .file_name("experiment.log")
            .init();
    }

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

#[cfg(feature = "fusion")]
impl<B> BackendDeviceInit for Fusion<B>
where
    B: FusionBackend + BackendDeviceInit,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        <B as BackendDeviceInit>::init_devices(train_cfg_builder)
    }
}

#[cfg(feature = "cubecl")]
impl<R, F, I, BT> BackendDeviceInit for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    R::Device: CubeDevice + Any,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        let seed = train_cfg_builder.get_random_seed().unwrap();
        let requested = train_cfg_builder.get_num_devices_per_node().unwrap();

        #[cfg(feature = "wgpu")]
        let is_wgpu = TypeId::of::<R::Device>() == TypeId::of::<WgpuDevice>();
        #[cfg(not(feature = "wgpu"))]
        let is_wgpu = false;

        let device_ids = if is_wgpu {
            if requested <= 1 {
                vec![DeviceId::new(4, 0)]
            } else {
                let available_discrete = <R::Device as CubeDevice>::device_count(0);
                let available_integrated = <R::Device as CubeDevice>::device_count(1);
                let available_virtual = <R::Device as CubeDevice>::device_count(2);
                let available_cpu = <R::Device as CubeDevice>::device_count(3);

                let type_id = if available_discrete >= requested {
                    0
                } else if available_integrated >= requested {
                    1
                } else if available_virtual >= requested {
                    2
                } else if available_cpu >= requested {
                    3
                } else {
                    panic!(
                        "Requested {requested} WGPU devices, but only found \
                         discrete={available_discrete}, integrated={available_integrated}, \
                         virtual={available_virtual}, cpu={available_cpu}. Reduce \
                         num_devices_per_node or change backend."
                    );
                };

                (0..requested)
                    .map(|i| DeviceId::new(type_id, i as u32))
                    .collect()
            }
        } else {
            let available = <R::Device as CubeDevice>::device_count_total();
            let count = requested.min(available.max(1));
            (0..count).map(|i| DeviceId::new(0, i as u32)).collect()
        };

        let devices = device_ids
            .into_iter()
            .map(<R::Device as CubeDevice>::from_id)
            .collect::<Vec<_>>();

        for device in &devices {
            #[cfg(feature = "wgpu")]
            if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
                #[cfg(feature = "metal")]
                burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                    wgpu_device,
                    Default::default(),
                );
                #[cfg(not(feature = "metal"))]
                burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                    wgpu_device,
                    Default::default(),
                );
            }

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
        wandb_logger = Some(init_logger_blocking(config));
        info!("Wandb logger initialized.");
    }
    wandb_logger
}

pub fn init_wandb_metric_logger() -> Option<WandbLogger> {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let mut wandb_logger: Option<WandbLogger> = None;
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
        wandb_logger = Some(init_metric_logger_blocking(config));
        info!("Wandb metric logger initialized.");
    }
    wandb_logger
}

pub fn init_renderer() -> (Interrupter, Box<dyn MetricsRenderer>) {
    let interrupter = Interrupter::new();
    if TRAIN_CFG.get().unwrap().use_tui {
        #[cfg(feature = "tui")]
        {
            return (
                interrupter.clone(),
                Box::new(TuiMetricsRenderer::new(interrupter, None)),
            );
        }
        #[cfg(not(feature = "tui"))]
        {
            warn!("use_tui=true but feature \"tui\" is disabled, falling back to bar renderer");
        }
    } else {
        return (
            interrupter,
            Box::new(BarMetricsRenderer::new(
                TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
            )),
        );
    }
    (
        interrupter,
        Box::new(BarMetricsRenderer::new(
            TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
        )),
    )
}
