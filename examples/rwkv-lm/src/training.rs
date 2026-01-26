// This module trains a next toke prediction language model using the provided training datasets,
// as well as the provided configuration. It first initializes batchers for the datasets,
// then initializes the model and data loaders for the datasets. The function then initializes
// an optimizer and a learning rate scheduler, and uses them along with the model and datasets
// to build a learner, which is used to train the model. The trained model and the configuration are
// then saved to the specified directory.

use log::info;
use rwkv::config::validated::model::{FinalModelConfigBuilder, MODEL_CFG};
use rwkv::config::validated::train::{FinalTrainConfigBuilder, TRAIN_CFG};
#[cfg(feature = "ddp")]
use rwkv::custom::collective::{AllReduceStrategy, CollectiveConfig};
use rwkv::custom::train::{Learner, SupervisedTraining};
#[cfg(not(feature = "ddp"))]
use rwkv::custom::{
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::{attention::SeqLengthOption, transformer::TransformerEncoderConfig},
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        MultiDeviceOptim,
        metric::{
            AccuracyMetric, CudaMetric, IterationSpeedMetric, LearningRateMetric, LossMetric,
        },
    },
};
use rwkv::custom::optim::LearningRate;
use rwkv::custom::tensor::backend::AutodiffBackend;
use rwkv::train::data::sliding::SlidingDataset;
use rwkv::train::optim::lr_scheduler::WsdLrSchedulerConfig;
use rwkv::train::optim::optimizer::GroupedOptimizerConfig;
use crate::model::AutoRegressiveModelConfig;

rwkv::custom_mode!();


// Define train function
pub fn train<B: AutodiffBackend>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset: SlidingDataset<u16>,
    model_cfg_builder: FinalModelConfigBuilder,
    mut train_cfg_builder: FinalTrainConfigBuilder,
    artifact_dir: &str,      // Directory to save model and config files
) {
    // let data_loaders = ;

    model_cfg_builder.build();
    train_cfg_builder.build();

    // Initialize model
    let mut model = AutoRegressiveModelConfig::new(
        MODEL_CFG.get().unwrap().num_cells,
        MODEL_CFG.get().unwrap().vocabulary_size,
        MODEL_CFG.get().unwrap().embedded_dim,
        MODEL_CFG.get().unwrap().num_heads,
        MODEL_CFG.get().unwrap().head_size_auto,
    ).init::<B>(&devices[0]);

    if TRAIN_CFG.get().unwrap().need_init_weight_auto {
        model.init_weights(&devices[0]);
        info!("Initializing model");
    }

    // Initialize optimizer
    let optim = GroupedOptimizerConfig::new(
        0.9, 0.99, 1e-16, TRAIN_CFG.get().unwrap().weight_decay
    ).init(&model);

    // Initialize learning rate scheduler
    let lr_scheduler = WsdLrSchedulerConfig::new(
        TRAIN_CFG.get().unwrap().learning_rate_start as LearningRate,
        TRAIN_CFG.get().unwrap().learning_rate_end as LearningRate,
        TRAIN_CFG.get().unwrap().warmup_steps,
        TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
    ).init();

    // Initialize learner
    #[cfg(not(feature = "ddp"))]
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train(IterationSpeedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .with_training_strategy(rwkv::custom::train::TrainingStrategy::MultiDevice(
            devices,
            MultiDeviceOptim::OptimSharded,
        ));

    #[cfg(feature = "ddp")]
    let collective_config =
        CollectiveConfig::default().with_local_all_reduce_strategy(AllReduceStrategy::Tree(2));
    #[cfg(feature = "ddp")]
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train(IterationSpeedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .with_training_strategy(rwkv::custom::train::ddp(devices, collective_config))
        .num_epochs(config.num_epochs)
        .summary();

    // Train the model
    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            result.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}