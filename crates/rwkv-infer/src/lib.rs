//! RWKV inference runtime (WIP).
//!
//! This crate provides:
//! - continuous-batching scheduling (prefill/decode);
//! - OpenAI-compatible HTTP server scaffolding (axum);
//! - in-memory "DB" utilities (response cache, background tasks).

pub mod api;
pub mod auth;
pub mod engine;
pub mod error;
pub mod init;
#[cfg(feature = "ipc-iceoryx2")]
pub mod ipc;
pub mod scheduler;
pub mod sdk;
pub mod server;
pub mod service;
pub mod storage;
#[cfg(feature = "trace")]
pub mod trace;
pub mod types;

pub use engine::{EngineCommand, EngineHandle};
pub use error::{Error, Result};
pub use server::{AppState, RouterBuilder};
pub use types::SamplingConfig;
