#[cfg(feature = "ipc")]
pub mod protocol;
#[cfg(feature = "ipc")]
mod server;

#[cfg(feature = "ipc")]
pub use protocol::{
    HandshakeRequest,
    HandshakeResponse,
    IPC_PROTOCOL_VERSION,
    IpcError,
    IpcRequest,
    IpcResponse,
    ResponseKind,
    RouteId,
};
#[cfg(feature = "ipc")]
pub use server::{IpcServer, IpcServerConfig};
