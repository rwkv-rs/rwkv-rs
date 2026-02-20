//! RWKV benchmark and profiling library.

pub mod error;
pub mod metrics;
pub mod profile;
pub mod report;
pub mod serving;

pub use error::{BenchError, Result};
