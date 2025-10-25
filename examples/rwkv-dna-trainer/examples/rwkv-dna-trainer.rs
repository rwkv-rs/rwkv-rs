use std::sync::Arc;

use burn::{
    backend::Cuda, data::dataloader::DataLoader, lr_scheduler::LrScheduler, optim::LearningRate,
};
use rwkv_config::validated::{
    model::MODEL_CFG,
    train::{FinalTrainConfigBuilder, TRAIN_CFG},
};
use rwkv_lm::auto_regressive_model::{AutoRegressiveModel, AutoRegressiveModelConfig};
use rwkv_train_cli::{
    data::sliding::{AutoRegressiveBatch, get_sliding_data_loaders},
    optim::{
        loss_fn::{AutoRegressiveLoss, AutoRegressiveLossConfig},
        lr_scheduler::{WsdLrScheduler, WsdLrSchedulerConfig},
        optimizer::{GroupedOptimizer, GroupedOptimizerConfig},
    },
    trainer::{
        RwkvTrainBackend,
        custom::{TrainStepContext, TrainStepOutput, Trainer},
    },
};

type MyBackend = Cuda<f32, i32>;

struct MyTrainer;

impl Trainer for MyTrainer {
    type Optimizer<B: RwkvTrainBackend> = GroupedOptimizer<B, Self::Model<B>>;
    type LrScheduler = WsdLrScheduler;
    type Loss<B: RwkvTrainBackend> = AutoRegressiveLoss<B>;
    type Batch<B: RwkvTrainBackend> = AutoRegressiveBatch<B>;
    type Model<B: RwkvTrainBackend> = AutoRegressiveModel<B>;
    type TokenUnit = u8;

    fn get_data_loaders<B: RwkvTrainBackend>(
        mut train_cfg_builder: &mut FinalTrainConfigBuilder,
        devices: &Vec<B::Device>,
    ) -> Vec<Arc<dyn DataLoader<B, Self::Batch<B>>>> {
        get_sliding_data_loaders::<B, Self::TokenUnit>(&mut train_cfg_builder, devices)
    }

    fn get_model<B: RwkvTrainBackend>(main_device: &B::Device) -> Self::Model<B> {
        AutoRegressiveModelConfig::new(
            MODEL_CFG.get().unwrap().num_cells,
            MODEL_CFG.get().unwrap().vocabulary_size,
            MODEL_CFG.get().unwrap().embedded_dim,
            MODEL_CFG.get().unwrap().num_heads,
            MODEL_CFG.get().unwrap().head_size_auto,
        )
        .init::<B, Self::TokenUnit>(main_device)
    }

    fn init_weight<B: RwkvTrainBackend>(main_model: &mut Self::Model<B>, main_device: &B::Device) {
        main_model.init_weights(main_device);
    }

    fn init_loss<B: RwkvTrainBackend>(device: &B::Device) -> Self::Loss<B> {
        AutoRegressiveLossConfig::new(
            TRAIN_CFG.get().unwrap().batch_size_per_device,
            TRAIN_CFG.get().unwrap().context_length,
            MODEL_CFG.get().unwrap().vocabulary_size,
        )
        .init::<B, Self::TokenUnit>(device)
    }

    fn init_optimizer<B: RwkvTrainBackend>(model: &Self::Model<B>) -> Self::Optimizer<B> {
        GroupedOptimizerConfig::new(0.9, 0.99, 1e-16, TRAIN_CFG.get().unwrap().weight_decay)
            .init(model)
    }

    fn init_lr_scheduler() -> Self::LrScheduler {
        WsdLrSchedulerConfig::new(
            TRAIN_CFG.get().unwrap().learning_rate_start as LearningRate,
            TRAIN_CFG.get().unwrap().learning_rate_end as LearningRate,
            TRAIN_CFG.get().unwrap().warmup_steps,
            TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
        )
        .init()
    }

    fn train_step<B: RwkvTrainBackend>(
        model: &mut Self::Model<B>,
        batch: Self::Batch<B>,
        loss_fn: &Self::Loss<B>,
        lr_scheduler: &mut Self::LrScheduler,
        context: TrainStepContext,
        _is_rank0: bool,
    ) -> TrainStepOutput<B> {
        let (logits, _) = model.forward(batch.inputs, vec![]);

        let loss = loss_fn.forward(
            logits,
            &batch.targets,
            TRAIN_CFG.get().unwrap().enable_l2wrap,
        );

        let learning_rate = if context.is_accumulation_boundary() {
            Some(lr_scheduler.step())
        } else {
            None
        };

        TrainStepOutput::new(loss, learning_rate)
    }
}

fn main() {
    MyTrainer::run::<MyBackend>(
        "examples/rwkv-dna-trainer/config/model.toml",
        "examples/rwkv-dna-trainer/config/train.toml",
    );
}
