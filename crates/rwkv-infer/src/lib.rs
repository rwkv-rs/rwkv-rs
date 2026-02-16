//! RWKV inference runtime (WIP).
//!
//! This crate provides:
//! - continuous-batching scheduling (prefill/decode);
//! - OpenAI-compatible HTTP server scaffolding (axum);
//! - in-memory "DB" utilities (response cache, background tasks).

pub mod auth;
pub mod engine;
pub mod error;
pub mod init;
pub mod scheduler;
pub mod service;
pub mod sdk;
pub mod server;
pub mod storage;
pub mod types;

pub use engine::{EngineCommand, EngineHandle};
pub use error::{Error, Result};
pub use server::{RwkvInferApp, RwkvInferRouterBuilder};
pub use types::SamplingConfig;
