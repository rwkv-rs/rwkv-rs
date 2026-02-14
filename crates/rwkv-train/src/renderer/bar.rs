use std::sync::Arc;

use burn::train::{
    metric::{MetricDefinition, MetricId},
    renderer::{
        EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
        MetricsRendererEvaluation, MetricsRendererTraining, TrainingProgress,
    },
};
use indicatif::{ProgressBar, ProgressStyle};
use rwkv_config::validated::train::TRAIN_CFG;

const LOSS_METRIC_NAME: &str = "Loss";

const LEARNING_RATE_METRIC_NAME: &str = "Learning Rate";

const ITERATION_SPEED_METRIC_NAME: &str = "Iteration Speed";

/// Progress bar renderer for training metrics
pub struct BarMetricsRenderer {
    pb: ProgressBar,
    epoch_index: usize,
    num_epochs: usize,
    train_loss: f64,
    train_lr: f64,
    train_kilo_tokens_per_sec: f64,
    valid_loss: Option<f64>,
    metric_id_loss: MetricId,
    metric_id_learning_rate: MetricId,
    metric_id_iteration_speed: MetricId,
    tokens_per_step: f64,
}

impl BarMetricsRenderer {
    pub fn new(num_epochs: usize) -> Self {
        let pb = ProgressBar::new(100);

        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} | {msg}",
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        Self {
            pb,
            epoch_index: 0,
            num_epochs,
            train_loss: 0.0,
            train_lr: 0.0,
            train_kilo_tokens_per_sec: 0.0,
            valid_loss: None,
            metric_id_loss: MetricId::new(Arc::new(LOSS_METRIC_NAME.to_string())),
            metric_id_learning_rate: MetricId::new(Arc::new(LEARNING_RATE_METRIC_NAME.to_string())),
            metric_id_iteration_speed: MetricId::new(Arc::new(ITERATION_SPEED_METRIC_NAME.to_string())),
            // Each burn-train iteration corresponds to one device-local batch (even in multi-device mode),
            // so use the per-device batch size here. Iteration Speed already scales with number of devices.
            tokens_per_step: (TRAIN_CFG.get().unwrap().context_length
                * TRAIN_CFG.get().unwrap().batch_size_per_device) as f64,
        }
    }
}

impl MetricsRendererTraining for BarMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        if let MetricState::Numeric(entry, value) = state {
            if entry.metric_id == self.metric_id_loss {
                self.train_loss = value.current();
            } else if entry.metric_id == self.metric_id_learning_rate {
                self.train_lr = value.current();
            } else if entry.metric_id == self.metric_id_iteration_speed {
                // Iteration Speed is in iter/sec. Convert to kilo-tokens/sec using per-device tokens
                // (Iteration Speed already scales with number of devices in multi-device mode).
                self.train_kilo_tokens_per_sec = value.current() * self.tokens_per_step / 1000.0;
            }
        }
    }

    fn update_valid(&mut self, state: MetricState) {
        if let MetricState::Numeric(entry, value) = state
            && entry.metric_id == self.metric_id_loss
        {
            self.valid_loss = Some(value.current());
        }
    }

    fn render_train(&mut self, item: TrainingProgress) {
        // Reset progress bar when epoch changes
        if item.epoch != self.epoch_index {
            self.epoch_index = item.epoch;
            self.pb.reset();
        }

        // burn-train increments `iteration` per device-local batch in multi-device mode.
        // Match the progress bar length to that convention.
        let cfg = TRAIN_CFG.get().unwrap();
        self.pb.set_length((cfg.num_steps_per_mini_epoch_auto * cfg.num_devices_per_node) as u64);
        self.pb.set_position(item.iteration as u64);
        self.pb.set_message(format!(
            "Epoch {}/{} | lr {:.2e} | kt/s {} | train_loss {:.5} | valid_loss {}",
            self.epoch_index,
            self.num_epochs,
            self.train_lr,
            if self.train_kilo_tokens_per_sec > 0.0 {
                format!("{:.2}", self.train_kilo_tokens_per_sec)
            } else {
                "-".to_string()
            },
            self.train_loss,
            self.valid_loss
                .map(|value| format!("{value:.5}"))
                .unwrap_or_else(|| "-".to_string()),
        ));
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        let message = match self.valid_loss {
            Some(loss) => format!(
                "Epoch {}/{} | valid_loss {:.5}",
                item.epoch,
                self.num_epochs,
                loss
            ),
            None => format!(
                "Epoch {}/{} | valid_loss -",
                item.epoch,
                self.num_epochs
            ),
        };

        self.pb.println(message);
    }
}

impl MetricsRendererEvaluation for BarMetricsRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}

    fn render_test(&mut self, _item: EvaluationProgress) {}
}

impl MetricsRenderer for BarMetricsRenderer {
    fn manual_close(&mut self) {
        self.pb.finish_with_message("Training completed");
    }

    fn register_metric(&mut self, _definition: MetricDefinition) {}
}
