//! Kernel 入口统一暴露 `rwkv-nn` 中需要特殊后端优化的算子.

pub mod addcmul;
pub(crate) mod backend;
pub mod guided_token_mask;
pub mod l2wrap;
pub mod rapid_sample;
pub mod token_shift_diff;
pub mod wkv7_common;
pub mod wkv7_infer;
pub mod wkv7_pretrain;
pub mod wkv7_statepass;
pub mod wkv7_statetune;
