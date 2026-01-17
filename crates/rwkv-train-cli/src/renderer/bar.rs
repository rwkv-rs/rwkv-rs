use std::sync::Arc;

use burn::train::{
    metric::{MetricDefinition, MetricId},
    renderer::{
        EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
        MetricsRendererEvaluation, MetricsRendererTraining, TrainingProgress,
    },
};
use indicatif::{ProgressBar, ProgressStyle};

const LOSS_METRIC_NAME: &str = "Loss";

const LEARNING_RATE_METRIC_NAME: &str = "Learning Rate";

/// Progress bar renderer for training metrics
pub struct BarMetricsRenderer {
    pb: ProgressBar,
    current_epoch: usize,
    total_epochs: usize,
    train_loss: f64,
    train_lr: f64,
    valid_loss: Option<f64>,
    metric_id_loss: MetricId,
    metric_id_learning_rate: MetricId,
}

impl BarMetricsRenderer {
    pub fn new(total_epochs: usize) -> Self {
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
            current_epoch: 0,
            total_epochs,
            train_loss: 0.0,
            train_lr: 0.0,
            valid_loss: None,
            metric_id_loss: MetricId::new(Arc::new(LOSS_METRIC_NAME.to_string())),
            metric_id_learning_rate: MetricId::new(Arc::new(LEARNING_RATE_METRIC_NAME.to_string())),
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
        assert!(
            item.progress.items_total > 0,
            "items_total must be positive, got {}",
            item.progress.items_total
        );

        // Reset progress bar when epoch changes
        if item.epoch != self.current_epoch {
            self.current_epoch = item.epoch;

            self.pb.reset();
        }

        // Update progress
        let total = item.progress.items_total;

        let current = item.progress.items_processed;

        let percentage = (current * 100) / total;

        self.pb.set_length(100);

        self.pb.set_position(percentage as u64);

        let valid_display = self
            .valid_loss
            .map(|value| format!("{value:.5}"))
            .unwrap_or_else(|| "-".to_string());

        self.pb.set_message(format!(
            "Epoch {}/{} | lr {:.2e} | train_loss {:.5} | valid_loss {}",
            self.current_epoch + 1,
            self.total_epochs,
            self.train_lr,
            self.train_loss,
            valid_display
        ));
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        let message = match self.valid_loss {
            Some(loss) => format!(
                "Epoch {}/{} | valid_loss {:.5}",
                item.epoch + 1,
                self.total_epochs,
                loss
            ),
            None => format!(
                "Epoch {}/{} | valid_loss -",
                item.epoch + 1,
                self.total_epochs
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
