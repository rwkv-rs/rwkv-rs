use serde::Deserialize;

use crate::ModelTypeOptions;

#[derive(Clone, Debug, Deserialize)]

pub struct RawModelConfig {
    pub model_type: ModelTypeOptions,

    pub num_cells: usize,
    pub vocab_size: usize,
    pub embedded_dim: usize,
    pub num_heads: usize,
    pub channel_mix_dim_scale: Option<usize>,
    pub dropout_prob: f64,

    pub with_token_shift: Option<bool>,
    pub with_deep_embed_att: Option<bool>,
    pub with_deep_embed_ffn: Option<bool>,
}

impl RawModelConfig {
    pub fn fill_default(&mut self) {
        if self.channel_mix_dim_scale.is_none() {
            self.channel_mix_dim_scale = Some(4);
        }

        if self.with_token_shift.is_none() {
            self.with_token_shift = Some(true);
        }

        if self.with_deep_embed_att.is_none() {
            self.with_deep_embed_att = Some(false);
        }

        if self.with_deep_embed_ffn.is_none() {
            self.with_deep_embed_ffn = Some(false);
        }
    }
}
