use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::{OptimizerOptions, TrainStageOptions};

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::train::RawTrainConfig", cell = "TRAIN_CFG")]
pub struct FinalTrainConfig {
    pub experiment_log_base_path: Option<String>,
    pub experiment_name: String,
    pub record_path: Option<String>,
    pub random_seed: u64,
    pub save_freq: usize,
    #[skip_raw]
    pub need_init_weight_auto: bool,

    pub dataset_base_path: String,
    pub filename_without_extensions: String,
    #[skip_raw]
    pub mmap_num_tokens_auto: usize,
    #[skip_raw]
    pub mmap_num_units_per_token: usize,
    #[skip_raw]
    pub mmap_token_dtype_auto: MmapTokenDtypeOptions,

    pub train_stage: TrainStageOptions,

    pub num_nodes: usize,
    pub num_devices_per_node: usize,
    pub batch_size_per_device: usize,
    #[skip_raw]
    pub batch_size_auto: usize,
    pub grad_checkpoint: bool,

    pub num_dataset_repeats: usize,
    pub context_length: usize,

    #[skip_raw]
    pub num_mini_epochs_auto: usize,
    #[skip_raw]
    pub num_steps_per_mini_epoch_auto: usize,
    #[skip_raw]
    pub magic_prime_auto: usize,

    pub optimizer: OptimizerOptions,
    pub learning_rate_start: f32,
    pub learning_rate_end: f32,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub gradient_clip_val: f32,
    pub num_accumulation_steps_per_device: usize,
    pub enable_l2wrap: bool,

    pub level: String,
    pub use_tui: bool,
    pub upload_to_wandb: bool,

    #[serde(skip_serializing)]
    pub wandb_api_key: Option<String>,
    pub wandb_entity_name: Option<String>,
    pub wandb_project_name: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub enum MmapTokenDtypeOptions {
    U8,
    U16,
    F32,
}

impl MmapTokenDtypeOptions {
    pub fn is_discrete(&self) -> bool {
        matches!(self, MmapTokenDtypeOptions::U8 | MmapTokenDtypeOptions::U16)
    }
}

impl FinalTrainConfigBuilder {
    pub fn fill_auto_after_load(&mut self) {
        let batch_size_auto = self.num_nodes.unwrap()
            * self.num_devices_per_node.unwrap()
            * self.batch_size_per_device.unwrap();
        let num_steps_per_mini_epoch_auto = 40320 / batch_size_auto;

        self.set_batch_size_auto(Some(batch_size_auto))
            .set_num_steps_per_mini_epoch_auto(Some(num_steps_per_mini_epoch_auto));
    }

    pub fn fill_after_read_record_file(&mut self, record_path: Option<String>) {
        self.set_record_path(record_path);
        if self.record_path.is_none() {
            self.set_need_init_weight_auto(Some(true));
        }
    }

    pub fn fill_after_read_bin(
        &mut self,
        mmap_num_tokens_auto: usize,
        mmap_num_units_per_token: usize,
        mmap_token_dtype_auto: MmapTokenDtypeOptions,
        magic_prime_auto: usize,
    ) {
        let num_mini_epochs_auto = self.num_dataset_repeats.unwrap() * mmap_num_tokens_auto
            / 40320
            / self.context_length.unwrap();

        assert!(
            (mmap_num_units_per_token == 1 && mmap_token_dtype_auto.is_discrete())
                || (mmap_num_units_per_token != 1 && !mmap_token_dtype_auto.is_discrete())
        );

        self.set_mmap_num_tokens_auto(Some(mmap_num_tokens_auto))
            .set_mmap_num_units_per_token(Some(mmap_num_units_per_token))
            .set_mmap_token_dtype_auto(Some(mmap_token_dtype_auto))
            .set_num_mini_epochs_auto(Some(num_mini_epochs_auto))
            .set_magic_prime_auto(Some(magic_prime_auto));
    }

    pub fn check(&self) {
        if self.get_num_nodes().unwrap() > 1 {
            panic!("Multiple nodes training are not supported yet");
        }
    }
}

pub static TRAIN_CFG: OnceCell<Arc<FinalTrainConfig>> = OnceCell::new();
