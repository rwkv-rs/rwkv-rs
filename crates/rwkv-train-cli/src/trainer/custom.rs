#![allow(clippy::type_complexity)]
#[cfg(feature = "hotpath")]
use std::sync::Barrier;
use std::{
    path::Path,
    sync::{Arc, atomic::Ordering, mpsc},
};

use async_trait::async_trait;
use burn::{
    backend::autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing},
    collective::{self, CollectiveConfig, PeerId},
    data::dataloader::{DataLoader, Progress},
    module::{AutodiffModule, Module, ModuleVisitor, Param},
    optim::LearningRate,
    prelude::*,
    train::{
        logger::Logger,
        metric::{MetricEntry, MetricId, NumericEntry, SerializedEntry},
        renderer::{MetricState, TrainingProgress},
    },
};
use burn_optim::{GradientsParams, Optimizer, lr_scheduler::LrScheduler};
use log::info;
#[cfg(feature = "hotpath")]
use rwkv_bench::HpScopeGuard;
use rwkv_config::validated::{
    model::FinalModelConfigBuilder,
    train::{FinalTrainConfigBuilder, TRAIN_CFG},
};
use rwkv_data::mmap::dtype::TokenUnit;
use rwkv_lm::kernels::wkv7::Wkv7Backend;
use wandb::LogData;

use crate::{
    data::EPOCH_INDEX,
    renderer::{METRIC_NAME_LEARNING_RATE, METRIC_NAME_LOSS, TrainMetricMessage},
    trainer::{
        RwkvAutodiff, RwkvTrainBackend,
        common::{
            init_cfg, init_devices, init_file_logger, init_log, init_renderer, init_wandb_logger,
        },
        ddp::GradSyncer,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct TrainStepContext {
    accumulation_step: usize,
    accumulation_limit: usize,
}

impl TrainStepContext {
    pub fn new(accumulation_step: usize, accumulation_limit: usize) -> Self {
        Self {
            accumulation_step,
            accumulation_limit: accumulation_limit.max(1),
        }
    }

    pub fn is_accumulation_boundary(&self) -> bool {
        (self.accumulation_step + 1).is_multiple_of(self.accumulation_limit)
    }
}

#[derive(Clone)]
pub struct TrainStepOutput<B: RwkvTrainBackend> {
    pub loss: Tensor<B, 1>,
    pub learning_rate: Option<LearningRate>,
}

impl<B: RwkvTrainBackend> TrainStepOutput<B> {
    pub fn new(loss: Tensor<B, 1>, learning_rate: Option<LearningRate>) -> Self {
        Self {
            loss,
            learning_rate,
        }
    }
}

#[async_trait]
pub trait Trainer {
    type Optimizer<B: RwkvTrainBackend>: Optimizer<Self::Model<B>, B> + Send;
    type LrScheduler: LrScheduler + Send;
    type Loss<B: RwkvTrainBackend>: Send + Sync;
    type Batch<B: RwkvTrainBackend>: Send + 'static;
    type Model<B: RwkvTrainBackend>: AutodiffModule<B> + Module<B> + Clone + Send + 'static;
    type TokenUnit: TokenUnit + Send + Sync;

    fn run<B>(model_cfg_path: &str, train_cfg_path: &str)
    where
        B: Backend + Wkv7Backend,
    {
        let (model_cfg_builder, mut train_cfg_builder) = init_cfg(model_cfg_path, train_cfg_path);

        let exp_log_path = init_log(&mut train_cfg_builder);

        if train_cfg_builder.get_grad_checkpoint().unwrap() {
            Self::train::<RwkvAutodiff<B, BalancedCheckpointing>>(
                model_cfg_builder,
                train_cfg_builder,
                &exp_log_path,
            );
        } else {
            Self::train::<RwkvAutodiff<B, NoCheckpointing>>(
                model_cfg_builder,
                train_cfg_builder,
                &exp_log_path,
            );
        }
    }

    fn train<B: RwkvTrainBackend>(
        model_cfg_builder: FinalModelConfigBuilder,
        mut train_cfg_builder: FinalTrainConfigBuilder,
        full_experiment_log_path: &Path,
    ) {
        let devices = init_devices::<B>(&train_cfg_builder);

        let data_loaders = Self::get_data_loaders::<B>(&mut train_cfg_builder, &devices);

        model_cfg_builder.build();
        train_cfg_builder.build();

        let mut main_model: Self::Model<B> = Self::get_model(&devices[0]);

        if TRAIN_CFG.get().unwrap().need_init_weight_auto {
            Self::init_weight(&mut main_model, &devices[0]);
            info!("Initializing model");
        }

        let mut file_logger = init_file_logger(full_experiment_log_path);
        let mut wandb_logger = init_wandb_logger();

        let (interrupter, mut renderer) = init_renderer();

        let (metrics_sender, metrics_receiver) = mpsc::channel::<TrainMetricMessage>();

        #[cfg(feature = "hotpath")]
        let mini_epoch_barrier =
            Arc::new(Barrier::new(TRAIN_CFG.get().unwrap().num_devices_per_node));

        let handles = {
            let metrics_sender_for_threads = metrics_sender.clone();

            devices
                .into_iter()
                .zip(data_loaders)
                .enumerate()
                .map(|(device_index, (device, loader))| {
                    let main_model = main_model.clone();

                    let metrics_sender = if device_index == 0 {
                        Some(metrics_sender_for_threads.clone())
                    } else {
                        None
                    };

                    let interrupter = interrupter.clone();

                    #[cfg(feature = "hotpath")]
                    let barrier_for_thread = Arc::clone(&mini_epoch_barrier);

                    std::thread::spawn(move || {
                        println!("[{device_index}] Running on device {device:?}");

                        let mut model = main_model.clone().fork(&device);

                        let syncer = GradSyncer::start::<B>(
                            CollectiveConfig::default()
                                .with_num_devices(TRAIN_CFG.get().unwrap().num_devices_per_node)
                                .with_local_all_reduce_strategy(
                                    collective::AllReduceStrategy::Tree(2),
                                ),
                            device.clone(),
                            PeerId::from(device_index),
                        );

                        let mut lr_scheduler = Self::init_lr_scheduler();
                        let mut optimizer = Self::init_optimizer::<B>(&model);
                        let loss_fn = Self::init_loss::<B>(&device);

                        let accumulation_steps = TRAIN_CFG
                            .get()
                            .unwrap()
                            .num_accumulation_steps_per_device
                            .max(1);

                        let mut accumulation_index = 0usize;
                        let mut accumulated_grads: Option<GradientsParams> = None;

                        let mut should_stop = false;
                        let is_rank0 = device_index == 0;

                        for mini_epoch_index in 0..TRAIN_CFG.get().unwrap().num_mini_epochs_auto {
                            #[cfg(feature = "hotpath")]
                            let _hp_mini_epoch_guard = {
                                let guard_label = format!("mini_epoch_{mini_epoch_index}");
                                HpScopeGuard::rank0(guard_label, is_rank0)
                            };

                            #[cfg(feature = "hotpath")]
                            barrier_for_thread.wait();

                            if interrupter.should_stop() {
                                should_stop = true;
                            }

                            if !should_stop {
                                EPOCH_INDEX.store(mini_epoch_index as u64, Ordering::Relaxed);

                                let batches = rwkv_bench::hp_iter_if(
                                    "data.next_batch",
                                    loader.iter(),
                                    is_rank0,
                                );

                                for (step_index, batch) in batches.enumerate() {
                                    if interrupter.should_stop() {
                                        should_stop = true;
                                        break;
                                    }

                                    let context = TrainStepContext::new(
                                        accumulation_index,
                                        accumulation_steps,
                                    );

                                    let mut output = Self::train_step::<B>(
                                        &mut model,
                                        batch,
                                        &loss_fn,
                                        &mut lr_scheduler,
                                        context,
                                        is_rank0,
                                    );

                                    let mut loss_for_backward = output.loss.clone();
                                    if accumulation_steps > 1 {
                                        let scale = 1.0 / accumulation_steps as f32;
                                        loss_for_backward = loss_for_backward * scale;
                                    }

                                    let grads_raw = loss_for_backward.backward();
                                    let grads_current =
                                        GradientsParams::from_grads(grads_raw, &model);

                                    if context.is_accumulation_boundary() {
                                        let total_grads = if let Some(existing) =
                                            accumulated_grads.take()
                                        {
                                            merge_gradients::<B, _>(&model, existing, grads_current)
                                        } else {
                                            grads_current
                                        };

                                        let synced_grads = syncer.sync(total_grads);
                                        let lr_value = output
                                            .learning_rate
                                            .take()
                                            .unwrap_or_else(|| lr_scheduler.step());

                                        if let Some(grads) = synced_grads {
                                            model = optimizer.step(lr_value, model, grads);
                                        }
                                        output.learning_rate = Some(lr_value);

                                        emit_metrics(
                                            &metrics_sender,
                                            mini_epoch_index,
                                            step_index,
                                            &output,
                                        );

                                        accumulation_index = 0;
                                    } else {
                                        accumulated_grads = Some(
                                            if let Some(existing) = accumulated_grads.take() {
                                                merge_gradients::<B, _>(
                                                    &model,
                                                    existing,
                                                    grads_current,
                                                )
                                            } else {
                                                grads_current
                                            },
                                        );

                                        emit_metrics(
                                            &metrics_sender,
                                            mini_epoch_index,
                                            step_index,
                                            &output,
                                        );

                                        accumulation_index += 1;
                                    }
                                }
                            }

                            #[cfg(feature = "hotpath")]
                            barrier_for_thread.wait();

                            if should_stop {
                                break;
                            }
                        }
                    })
                })
                .collect::<Vec<_>>()
        };

        drop(metrics_sender);

        let mut global_train_step_index = 0usize;

        let metric_id_loss = MetricId::new(Arc::new(METRIC_NAME_LOSS.to_string()));

        let metric_id_learning_rate = MetricId::new(Arc::new(METRIC_NAME_LEARNING_RATE.to_string()));

        while let Ok(message) = metrics_receiver.recv() {
            global_train_step_index += 1;

            let progress = TrainingProgress {
                progress: Progress {
                    items_processed: (message.step_in_epoch + 1)
                        .min(TRAIN_CFG.get().unwrap().num_steps_per_mini_epoch_auto),
                    items_total: TRAIN_CFG.get().unwrap().num_steps_per_mini_epoch_auto,
                },
                epoch: message.mini_epoch,
                epoch_total: TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
                iteration: global_train_step_index,
            };

            let loss_formatted = format!("{:.5}", message.loss);

            let loss_entry = MetricEntry::new(
                metric_id_loss.clone(),
                SerializedEntry::new(loss_formatted.clone(), loss_formatted),
            );

            renderer.update_train(MetricState::Numeric(
                loss_entry,
                NumericEntry::Value(message.loss),
            ));

            if let Some(lr) = message.learning_rate {
                let lr_formatted = format!("{lr:.2e}");

                let lr_entry = MetricEntry::new(
                    metric_id_learning_rate.clone(),
                    SerializedEntry::new(lr_formatted.clone(), lr_formatted),
                );

                renderer.update_train(MetricState::Numeric(lr_entry, NumericEntry::Value(lr)));
            }

            renderer.render_train(progress);

            if let Some(logger) = wandb_logger.as_mut() {
                let mut log = LogData::new();
                log.insert("_step", global_train_step_index as u64);
                log.insert("loss", message.loss);
                if let Some(lr) = message.learning_rate {
                    log.insert("lr", lr);
                }
                logger.log(log);
            }

            let lr_field = message
                .learning_rate
                .map(|value| format!("{value:.6e}"))
                .unwrap_or_default();

            let record = format!(
                "{},{},{},{},{:.5}",
                global_train_step_index,
                message.mini_epoch,
                message.step_in_epoch,
                lr_field,
                message.loss,
            );

            file_logger.log(record);
        }

        renderer.on_train_end(None).ok();

        if !TRAIN_CFG.get().unwrap().use_tui {
            renderer.manual_close();
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    fn get_data_loaders<B: RwkvTrainBackend>(
        train_cfg_builder: &mut FinalTrainConfigBuilder,
        devices: &[B::Device],
    ) -> Vec<Arc<dyn DataLoader<B, Self::Batch<B>>>>;

    fn get_model<B: RwkvTrainBackend>(main_device: &B::Device) -> Self::Model<B>;

    fn init_weight<B: RwkvTrainBackend>(main_model: &mut Self::Model<B>, main_device: &B::Device);

    fn init_loss<B: RwkvTrainBackend>(device: &B::Device) -> Self::Loss<B>;

    fn init_optimizer<B: RwkvTrainBackend>(model: &Self::Model<B>) -> Self::Optimizer<B>;

    fn init_lr_scheduler() -> Self::LrScheduler;

    fn train_step<B: RwkvTrainBackend>(
        model: &mut Self::Model<B>,
        batch: Self::Batch<B>,
        loss_fn: &Self::Loss<B>,
        lr_scheduler: &mut Self::LrScheduler,
        context: TrainStepContext,
        is_rank0: bool,
    ) -> TrainStepOutput<B>;
}

fn emit_metrics<B: RwkvTrainBackend>(
    sender: &Option<mpsc::Sender<TrainMetricMessage>>,
    mini_epoch: usize,
    step_index: usize,
    output: &TrainStepOutput<B>,
) {
    if let Some(sender) = sender {
        let loss_value = tensor_to_f64(&output.loss);
        let lr_value = output.learning_rate;

        let message = TrainMetricMessage {
            mini_epoch,
            step_in_epoch: step_index,
            loss: loss_value,
            learning_rate: lr_value,
        };

        sender.send(message).ok();
    }
}

fn tensor_to_f64<B: RwkvTrainBackend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor.clone().into_scalar().elem::<f32>() as f64
}

fn merge_gradients<B, M>(
    model: &M,
    mut accumulated: GradientsParams,
    mut fresh: GradientsParams,
) -> GradientsParams
where
    B: RwkvTrainBackend,
    M: AutodiffModule<B>,
{
    struct Accumulator<'a, B: RwkvTrainBackend> {
        accumulated: &'a mut GradientsParams,
        fresh: &'a mut GradientsParams,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: RwkvTrainBackend> ModuleVisitor<B> for Accumulator<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(mut fresh_grad) = self.fresh.remove::<B::InnerBackend, D>(param.id) {
                if let Some(existing) = self.accumulated.remove::<B::InnerBackend, D>(param.id) {
                    fresh_grad = existing + fresh_grad;
                }

                self.accumulated
                    .register::<B::InnerBackend, D>(param.id, fresh_grad);
            }
        }
    }

    let mut visitor = Accumulator::<B> {
        accumulated: &mut accumulated,
        fresh: &mut fresh,
        _marker: std::marker::PhantomData,
    };

    model.visit(&mut visitor);

    accumulated
}
