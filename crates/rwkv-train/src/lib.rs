#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! `rwkv-train` 暴露训练期使用的数据, 优化与日志组件.

#[macro_use]
extern crate derive_new;

#[cfg(feature = "train")]
pub mod data;
#[cfg(feature = "train")]
pub mod learner;
#[cfg(feature = "train")]
pub mod logger;
#[cfg(feature = "train")]
pub mod optim;
#[cfg(feature = "train")]
pub mod renderer;
#[cfg(all(feature = "train", feature = "trace"))]
pub mod trace;
#[cfg(feature = "train")]
pub mod utils;
