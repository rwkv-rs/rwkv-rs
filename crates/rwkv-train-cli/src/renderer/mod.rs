mod bar;

pub use bar::BarMetricsRenderer;

pub const METRIC_NAME_LOSS: &str = "Loss";
pub const METRIC_NAME_LEARNING_RATE: &str = "Learning Rate";

pub struct TrainMetricMessage {
    pub mini_epoch: usize,
    pub step_in_epoch: usize,
    pub loss: f64,
    pub learning_rate: Option<f64>,
}
