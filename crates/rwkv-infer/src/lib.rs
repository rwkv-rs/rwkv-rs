//! RWKV inference runtime (WIP).
//!
//! This crate provides:
//! - continuous-batching scheduling (prefill/decode);
//! - OpenAI-compatible HTTP server scaffolding (axum);
//! - in-memory "DB" utilities (response cache, background tasks).

pub mod auth;
pub mod config;
pub mod engine;
pub mod error;
pub mod scheduler;
pub mod sdk;
pub mod server;
pub mod storage;
pub mod types;

pub use config::{BackendConfig, SamplingConfig};
pub use engine::{EngineCommand, EngineHandle};
pub use error::{Error, Result};
pub use server::{RwkvInferRouterBuilder, SharedRwkvInferState};
