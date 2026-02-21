//! RWKV benchmark and profiling library.

pub mod error;
pub mod metrics;
pub mod profile;
pub mod report;
pub mod serving;
pub mod trace;

pub use error::{BenchError, Result};

#[cfg(feature = "trace")]
#[doc(hidden)]
pub use tracy_client as __tracy_client;

/// Emit a Tracy span scoped to the current block/function.
///
/// This macro becomes a no-op when the `trace` feature is disabled.
#[cfg(feature = "trace")]
#[macro_export]
macro_rules! tracy_scope {
    ($name:literal) => {
        let _rwkv_bench_tracy_span_guard = $crate::__tracy_client::span!($name);
    };
}

#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! tracy_scope {
    ($name:literal) => {};
}
