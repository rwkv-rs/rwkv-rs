use serde::{Deserialize, Serialize};

use crate::{DatasetFormatOptions, OptimizerOptions, fill_default};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RawTrainConfig {
    pub experiment_log_base_path: Option<String>,
    pub experiment_name: String,
    pub record_path: Option<String>,
    pub random_seed: Option<u64>,
    pub save_freq: Option<usize>,

    pub dataset_base_path: String,
    pub filename_without_extensions: String,
    pub dataset_format: Option<DatasetFormatOptions>,

    pub num_nodes: Option<usize>,
    pub num_devices_per_node: Option<usize>,
    pub batch_size_per_device: Option<usize>,
    pub grad_checkpoint: Option<bool>,

    pub num_dataset_repeats: Option<usize>,
    pub context_length: Option<usize>,
    pub paragraph_length: Option<usize>,

    pub optimizer: OptimizerOptions,
    pub learning_rate_start: f32,
    pub learning_rate_end: f32,
    pub warmup_steps: usize,
    pub weight_decay: Option<f32>,
    pub gradient_clip_val: Option<f32>,
    pub num_accumulation_steps_per_device: Option<usize>,
    pub enable_l2wrap: Option<bool>,

    pub level: Option<String>,
    pub use_tui: Option<bool>,
    pub upload_to_wandb: Option<bool>,
    #[serde(skip_serializing)]
    pub wandb_api_key: Option<String>,
    pub wandb_entity_name: Option<String>,
    pub wandb_project_name: Option<String>,
}

impl RawTrainConfig {
    pub fn fill_default(&mut self) {
        fill_default!(self,
            experiment_log_base_path: "logs".to_string(),
            random_seed: 42,
            save_freq: 1,
            dataset_format: DatasetFormatOptions::Rwkv,
            num_nodes: 1,
            num_devices_per_node: 1,
            batch_size_per_device: 1,
            grad_checkpoint: true,
            context_length: 512,
            paragraph_length: 512,
            weight_decay: 1e-3,
            gradient_clip_val: 1.0,
            num_accumulation_steps_per_device: 1,
            enable_l2wrap: true,
            level: "warn".to_string(),
            use_tui: true,
            upload_to_wandb: false,
        );
        if self.upload_to_wandb.unwrap() {
            assert!(
                self.wandb_api_key.is_some() && self.wandb_project_name.is_some(),
                "Wandb API Key and Project Name is required."
            );
        } else if self.wandb_api_key.is_none() && self.wandb_project_name.is_none() {
            eprintln!("Warning: Upload_to_wandb is False but you set Wandb API Key and Project.");
        }
    }
}
