//! 可解释性与 tracing（WIP）。
//!
//! 该 crate 未来将提供激活/权重/上下文学习等方向的可解释性研究工具链。
//! Interpretability & tracing (WIP) for activations/weights/in-context learning analysis.

#[cfg(feature = "tracy")]
#[doc(hidden)]
pub use tracy_client as __tracy_client;

/// Emit a Tracy span scoped to the current block/function.
///
/// This macro becomes a no-op when the `tracy` feature is disabled.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! tracy_scope {
    ($name:literal) => {
        let _rwkv_tracy_span_guard = $crate::__tracy_client::span!($name);
    };
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! tracy_scope {
    ($name:literal) => {};
}
