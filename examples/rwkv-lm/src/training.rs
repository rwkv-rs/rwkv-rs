// This module trains a next toke prediction language model using the provided training datasets,
// as well as the provided configuration. It first initializes batchers for the datasets,
// then initializes the model and data loaders for the datasets. The function then initializes
// an optimizer and a learning rate scheduler, and uses them along with the model and datasets
// to build a learner, which is used to train the model. The trained model and the configuration are
// then saved to the specified directory.

use crate::data::batcher::AutoRegressiveBatcher;
use crate::model::AutoRegressiveModelConfig;
use log::info;
#[cfg(not(feature = "tui"))]
use log::warn;
use rwkv::config::validated::train::{FinalTrainConfigBuilder, TRAIN_CFG};
use rwkv::config::{
    validated::model::{FinalModelConfigBuilder, MODEL_CFG},
    DatasetFormatOptions,
};
#[cfg(feature = "ddp")]
use rwkv::custom::collective::{AllReduceStrategy, CollectiveConfig};
use rwkv::custom::data::dataloader::DataLoaderBuilder;
use rwkv::custom::optim::LearningRate;
use rwkv::custom::prelude::Module;
use rwkv::custom::record::{CompactRecorder, Recorder};
use rwkv::custom::tensor::backend::AutodiffBackend;
#[cfg(not(feature = "ddp"))]
use rwkv::custom::train::MultiDeviceOptim;
use rwkv::custom::train::{
    logger::FileMetricLogger,
    metric::{CudaMetric, IterationSpeedMetric, LearningRateMetric, LossMetric},
    Interrupter, Learner, SupervisedTraining,
};
use rwkv::data::mmap::sample::Sampler;
use rwkv::nn::kernels::l2wrap::L2WrapBackend;
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;
use rwkv::train::data::sliding::{MmapBinReader, SlidingDataset};
use rwkv::train::learner::init::init_wandb_metric_logger;
use rwkv::train::optim::lr_scheduler::WsdLrSchedulerConfig;
use rwkv::train::optim::optimizer::{GroupedOptimizerConfig, ParamGroupingMode};
use rwkv::train::renderer::BarMetricsRenderer;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

rwkv::custom_mode!();

// Define train function
pub fn train<B: AutodiffBackend>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    model_cfg_builder: FinalModelConfigBuilder,
    mut train_cfg_builder: FinalTrainConfigBuilder,
    exp_log_path: &Path, // Experiment log directory (also used for artifacts)
) where
    B: Wkv7Backend + L2WrapBackend,
    B::InnerBackend: Wkv7Backend + L2WrapBackend,
{
    let bin_path = PathBuf::from(train_cfg_builder.get_dataset_base_path().unwrap())
        .join(train_cfg_builder.get_filename_without_extensions().unwrap())
        .with_extension("bin");

    let dataset_format = train_cfg_builder
        .get_dataset_format()
        .unwrap_or(DatasetFormatOptions::Rwkv);
    let bin = Arc::new(MmapBinReader::<u16>::open(&bin_path, dataset_format));
    train_cfg_builder.fill_after_read_bin(
        bin.num_tokens() as usize,
        bin.num_units_per_token() as usize,
        bin.dtype(),
        bin.get_magic_prime(train_cfg_builder.get_context_length().unwrap() as u64) as usize,
    );

    let num_devices = devices.len();
    let mut samplers = Vec::with_capacity(num_devices);
    for device_index in 0..num_devices {
        let sampler = Sampler::new(
            train_cfg_builder.get_num_devices_per_node().unwrap() as u64,
            device_index as u64,
            (train_cfg_builder
                .get_num_steps_per_mini_epoch_auto()
                .unwrap()
                * train_cfg_builder.get_batch_size_auto().unwrap()) as u64,
            train_cfg_builder.get_magic_prime_auto().unwrap() as u64,
        );
        samplers.push(sampler);
    }

    let dataset = Arc::new(SlidingDataset::new(
        train_cfg_builder.get_context_length().unwrap() as u64,
        bin.clone(),
        samplers,
    ));

    let batcher = AutoRegressiveBatcher::<B, u16>::new(
        bin.clone(),
        train_cfg_builder.get_mmap_num_units_per_token().unwrap(),
        train_cfg_builder.get_batch_size_per_device().unwrap(),
        train_cfg_builder.get_context_length().unwrap(),
    );

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(train_cfg_builder.get_batch_size_per_device().unwrap())
        .num_workers(1)
        .build(dataset.clone());

    let batcher_valid = AutoRegressiveBatcher::<B::InnerBackend, u16>::new(
        bin.clone(),
        train_cfg_builder.get_mmap_num_units_per_token().unwrap(),
        train_cfg_builder.get_batch_size_per_device().unwrap(),
        train_cfg_builder.get_context_length().unwrap(),
    );
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(train_cfg_builder.get_batch_size_per_device().unwrap())
        .num_workers(1)
        .set_device(devices[0].clone())
        .build(dataset.clone());
    let dataloader_valid = dataloader_valid.slice(0, 0);

    model_cfg_builder.build();
    train_cfg_builder.build();

    // Initialize model
    let mut model = AutoRegressiveModelConfig::new(
        MODEL_CFG.get().unwrap().num_cells,
        MODEL_CFG.get().unwrap().vocabulary_size,
        MODEL_CFG.get().unwrap().embedded_dim,
        MODEL_CFG.get().unwrap().num_heads,
        MODEL_CFG.get().unwrap().head_size_auto,
    )
    .init::<B>(&devices[0]);

    if TRAIN_CFG.get().unwrap().need_init_weight_auto {
        model.init_weights(&devices[0]);
        info!("Initializing model");
    }

    // Initialize optimizer
    #[cfg(not(any(feature = "statetune")))]
    let optim =
        GroupedOptimizerConfig::new(0.9, 0.99, 1e-16, TRAIN_CFG.get().unwrap().weight_decay)
            .init(&model, ParamGroupingMode::Default);

    #[cfg(feature = "statetune")]
    let optim =
        GroupedOptimizerConfig::new(0.9, 0.99, 1e-16, TRAIN_CFG.get().unwrap().weight_decay)
            .init(&model, ParamGroupingMode::StateTune);

    // Initialize learning rate scheduler
    let lr_scheduler = WsdLrSchedulerConfig::new(
        TRAIN_CFG.get().unwrap().learning_rate_start as LearningRate,
        TRAIN_CFG.get().unwrap().learning_rate_end as LearningRate,
        TRAIN_CFG.get().unwrap().warmup_steps,
        TRAIN_CFG.get().unwrap().num_mini_epochs_auto
            * TRAIN_CFG.get().unwrap().num_steps_per_mini_epoch_auto,
    )
    .init();

    // Initialize learner
    let interrupter = Interrupter::new();

    let mut training = SupervisedTraining::new(exp_log_path, dataloader_train, dataloader_valid)
        .metric_train(CudaMetric::new())
        .metric_train_numeric(IterationSpeedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(TRAIN_CFG.get().unwrap().num_mini_epochs_auto)
        .summary()
        .with_interrupter(interrupter.clone())
        .with_application_logger(None);

    training = training.with_metric_logger(FileMetricLogger::new(exp_log_path));

    if let Some(wandb_logger) = init_wandb_metric_logger() {
        training = training.with_metric_logger(wandb_logger);
    }

    let training = if TRAIN_CFG.get().unwrap().use_tui {
        #[cfg(feature = "tui")]
        {
            training.renderer(rwkv::custom::train::renderer::tui::TuiMetricsRenderer::new(
                interrupter.clone(),
                None,
            ))
        }
        #[cfg(not(feature = "tui"))]
        {
            warn!("use_tui=true but feature \"tui\" is disabled, falling back to bar renderer");
            training.renderer(BarMetricsRenderer::new(
                TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
            ))
        }
    } else {
        training.renderer(BarMetricsRenderer::new(
            TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
        ))
    };

    #[cfg(not(feature = "ddp"))]
    let training = training.with_training_strategy(
        rwkv::custom::train::TrainingStrategy::MultiDevice(devices, MultiDeviceOptim::OptimSharded),
    );

    #[cfg(feature = "ddp")]
    let collective_config =
        CollectiveConfig::default().with_local_all_reduce_strategy(AllReduceStrategy::Tree(2));
    #[cfg(feature = "ddp")]
    let training =
        training.with_training_strategy(rwkv::custom::train::ddp(devices, collective_config));

    // Train the model
    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    // Save the configuration and the trained model
    let config_json = sonic_rs::json!({
        "model": MODEL_CFG.get().unwrap().as_ref(),
        "train": TRAIN_CFG.get().unwrap().as_ref(),
    });
    fs::write(
        exp_log_path.join("config.json"),
        sonic_rs::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    CompactRecorder::new()
        .record(result.model.into_record(), exp_log_path.join("model"))
        .unwrap();
}
