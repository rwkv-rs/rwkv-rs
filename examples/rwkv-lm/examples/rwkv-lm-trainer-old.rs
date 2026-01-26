use std::sync::Arc;
use rwkv::config::validated::model::MODEL_CFG;
use rwkv::config::validated::train::{FinalTrainConfigBuilder, TRAIN_CFG};
use rwkv::custom::backend::Cuda;
use rwkv::custom::data::dataloader::DataLoader;
use rwkv::custom::lr_scheduler::LrScheduler;
use rwkv::custom::optim::LearningRate;
use rwkv::train::optim::lr_scheduler::{WsdLrScheduler, WsdLrSchedulerConfig};
use rwkv::train::optim::optimizer::{GroupedOptimizer, GroupedOptimizerConfig};
use rwkv::train::trainer::custom::{TrainStepContext, TrainStepOutput, Trainer};
use rwkv::train::trainer::RwkvTrainBackend;
use rwkv_lm::model::{AutoRegressiveModel, AutoRegressiveModelConfig};

type MyBackend = Cuda<f32, i32>;

struct MyTrainer;

impl Trainer for MyTrainer {
    type Optimizer<B: RwkvTrainBackend> = GroupedOptimizer<B, Self::Model<B>>;
    type LrScheduler = WsdLrScheduler;
    type Loss<B: RwkvTrainBackend> = AutoRegressiveLoss<B>;
    type Batch<B: RwkvTrainBackend> = AutoRegressiveBatch<B>;
    type Model<B: RwkvTrainBackend> = AutoRegressiveModel<B>;
    type TokenUnit = u16;

    fn get_data_loaders<B: RwkvTrainBackend>(
        train_cfg_builder: &mut FinalTrainConfigBuilder,
        devices: &[B::Device],
    ) -> Vec<Arc<dyn DataLoader<B, Self::Batch<B>>>> {
        get_sliding_data_loaders::<B, Self::TokenUnit>(train_cfg_builder, devices)
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
        let (logits, _) = model.forward(batch.inputs);

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
        "examples/rwkv-lm/config/model.toml",
        "examples/rwkv-lm/config/train.toml",
    );
}
