use std::sync::atomic::AtomicU64;

pub mod sliding;

pub static EPOCH_INDEX: AtomicU64 = AtomicU64::new(0);
