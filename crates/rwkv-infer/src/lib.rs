//! RWKV inference runtime (WIP).
//!
//! This crate exposes a literal module layout:
//! - `access`: request ingress adapters (HTTP, IPC, local Rust API)
//! - `inference_core`: sampling, logprobs, request output, batching, execution loop, request state
//! - `model_pool`: loaded-model groups and reload management
//! - `response_store`: cached responses and background task state
//! - `config_loader`: infer config loading and path resolution

pub mod access;
pub mod auth;
pub mod config_loader;
pub mod error;
pub mod inference_core;
pub mod model_pool;
pub mod response_store;
#[cfg(feature = "trace")]
pub mod trace;

pub use access::http_api::{AppState, HttpApiRouterBuilder, HttpApiState, RouterBuilder};
#[cfg(feature = "ipc-iceoryx2")]
pub use access::ipc_api::{IpcClientConfig, IpcOpenAiClient, IpcServer, IpcServerConfig};
pub use access::local_api::{LocalInferenceClient, RwkvInferClient};
pub use error::{Error, Result};
pub use inference_core::{
    InferenceExecutionConfig, InferenceExecutionLoop, InferenceSubmitCommand,
    InferenceSubmitHandle, InferenceSubmitResult, ModelForward, SamplingConfig,
    TokenLogprobsConfig,
};
pub use model_pool::{
    LoadedModelGroup, LoadedModelRegistry, ModelEngineFactory, ModelRequestRouter,
};
