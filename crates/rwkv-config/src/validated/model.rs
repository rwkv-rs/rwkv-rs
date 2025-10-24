use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::ModelTypeOptions;

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::model::RawModelConfig", cell = "MODEL_CFG")]
pub struct FinalModelConfig {
    pub model_type: ModelTypeOptions,

    pub num_cells: usize,
    pub vocabulary_size: usize,
    pub embedded_dim: usize,
    pub num_heads: usize,
    pub channel_mix_dim_scale: usize,
    pub dropout_prob: f64,

    pub with_token_shift: bool,
    pub with_deep_embed_att: bool,
    pub with_deep_embed_ffn: bool,

    #[skip_raw]
    pub head_size_auto: usize,
}

impl FinalModelConfigBuilder {
    pub fn fill_auto_after_load(&mut self) {
        self.set_head_size_auto(Some(
            self.get_embedded_dim().unwrap() / self.get_num_heads().unwrap(),
        ));
    }
}

pub static MODEL_CFG: OnceCell<Arc<FinalModelConfig>> = OnceCell::new();
