pub mod http_client;
#[cfg(feature = "ipc")]
pub mod ipc_client;
pub mod local;

#[cfg(feature = "ipc")]
pub use ipc_client::{IpcClientConfig, IpcOpenAiClient};
pub use local::{LocalClient, RwkvInferClient};
