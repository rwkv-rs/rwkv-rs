//! Shared benchmarking and profiling helpers for RWKV components.
//!
//! This crate provides thin wrappers around the `hotpath` profiler so
//! downstream crates can instrument code without littering features or macros
//! everywhere.

/// Macro to measure an arbitrary code block when the `hotpath` feature is
/// enabled.
///
/// Usage:
/// ```ignore
/// let result = rwkv_bench::hp_block!("forward", || model.forward(input));
/// ```
#[macro_export]
macro_rules! hp_block {
    ($label:expr, $body:expr) => {{
        #[cfg(feature = "hotpath")]
        {
            hotpath::measure_block!($label, ($body)())
        }

        #[cfg(not(feature = "hotpath"))]
        {
            ($body)()
        }
    }};
}

/// Macro to measure a block only when the current worker participates (e.g.
/// rank 0).
///
/// Usage:
/// ```ignore
/// let grads = rwkv_bench::hp_rank0_block!(is_rank0, "ddp_sync", || syncer.sync(grads));
/// ```
#[macro_export]
macro_rules! hp_rank0_block {
    ($is_active:expr, $label:expr, $body:expr) => {{
        if $is_active {
            $crate::hp_block!($label, $body)
        } else {
            ($body)()
        }
    }};
}

/// Conditionally measure a code block only when `cond` evaluates to true.
#[macro_export]
macro_rules! hp_block_if {
    ($cond:expr, $label:expr, $body:expr) => {{
        if $cond {
            $crate::hp_block!($label, $body)
        } else {
            ($body)()
        }
    }};
}

/// Returns `true` when the build enables hotpath profiling.
#[inline]
pub const fn is_hotpath_enabled() -> bool {
    cfg!(feature = "hotpath")
}

/// RAII helper that starts a hotpath guard if profiling is active and enabled.
pub struct HpScopeGuard {
    #[cfg(feature = "hotpath")]
    guard: Option<hotpath::HotPath>,
}

impl HpScopeGuard {
    /// Create a guard with the provided label.
    ///
    /// The guard is only armed when both the `hotpath` feature is enabled and
    /// `active` evaluates to `true`.
    pub fn new(label: impl Into<String>, active: bool) -> Self {
        #[cfg(feature = "hotpath")]
        {
            const DEFAULT_PERCENTILES: &[u8] = &[50, 90, 99];

            let guard = if active {
                Some(
                    hotpath::GuardBuilder::new(label.into())
                        .percentiles(DEFAULT_PERCENTILES)
                        .build(),
                )
            } else {
                None
            };

            Self { guard }
        }

        #[cfg(not(feature = "hotpath"))]
        {
            let _ = label;

            let _ = active;

            Self {}
        }
    }

    /// Convenience helper for the common pattern where only rank 0 should emit
    /// data.
    pub fn rank0(label: impl Into<String>, is_rank0: bool) -> Self {
        Self::new(label, is_rank0)
    }

    /// Returns whether the inner guard is active.
    pub fn is_active(&self) -> bool {
        #[cfg(feature = "hotpath")]
        {
            self.guard.is_some()
        }

        #[cfg(not(feature = "hotpath"))]
        {
            false
        }
    }
}

impl Default for HpScopeGuard {
    fn default() -> Self {
        Self::new("inactive", false)
    }
}

/// Iterator wrapper that measures each `next` call with a specific label.
pub struct HpIter<I> {
    inner: I,
    active: bool,
    #[cfg(feature = "hotpath")]
    label: &'static str,
}

impl<I> Iterator for HpIter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.active {
            return self.inner.next();
        }

        #[cfg(feature = "hotpath")]
        {
            crate::hp_block!(self.label, || self.inner.next())
        }

        #[cfg(not(feature = "hotpath"))]
        {
            self.inner.next()
        }
    }
}

/// Wrap an iterator so that each `next` call is profiled with `hp_block!`.
#[inline]
pub fn hp_iter<I>(label: &'static str, iter: I) -> HpIter<I>
where
    I: Iterator,
{
    hp_iter_if(label, iter, true)
}

/// Same as [`hp_iter`] but only active when `active` is true.
#[inline]
pub fn hp_iter_if<I>(label: &'static str, iter: I, active: bool) -> HpIter<I>
where
    I: Iterator,
{
    #[cfg(feature = "hotpath")]
    {
        HpIter {
            inner: iter,
            active,
            label,
        }
    }

    #[cfg(not(feature = "hotpath"))]
    {
        let _ = label;

        HpIter {
            inner: iter,
            active,
        }
    }
}
