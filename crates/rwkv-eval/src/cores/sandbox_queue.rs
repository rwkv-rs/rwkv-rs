use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct SandboxVerdict {
    pub passed: bool,
    pub fail_reason: String,
    pub stdout: String,
    pub stderr: String,
}

pub type SandboxQueue = mpsc::Sender<SandboxQueueRequest>;

pub struct SandboxQueueRequest {
    pub script: String,
    pub result_tx: oneshot::Sender<Result<SandboxVerdict, String>>,
}
