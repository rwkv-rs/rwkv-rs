use core::fmt::Debug;
use std::{path::PathBuf, sync::Arc, thread, time::Duration};

use iceoryx2::{
    active_request::ActiveRequest,
    prelude::{NodeBuilder, ZeroCopySend, ipc},
};
use tokio::runtime::Handle;
use uuid::Uuid;

use crate::{
    dtos::{
        admin::models::reload::{ModelsReloadReq, ModelsReloadResp},
        chat::completions::{ChatCompletionResp, ChatCompletionsReq},
        completions::{CompletionsReq, CompletionsResp},
    },
    handlers::auth::{AuthConfig, check_api_key},
    services::{QueueMapBuilder, SharedQueueMap, health::GpuMetricsCache, select_model_queue},
};
use super::protocol::{
    HandshakeRequest,
    HandshakeResponse,
    IPC_PROTOCOL_VERSION,
    IpcError,
    IpcRequest,
    IpcResponse,
    IpcResult,
    ResponseKind,
    RouteId,
    decode_json,
    decode_request,
    encode_json,
    encode_response,
};

#[derive(Clone, Debug)]
pub struct IpcServerConfig {
    pub service_name: String,
    pub max_request_bytes: usize,
    pub max_response_bytes: usize,
    pub max_inflight_requests: usize,
    pub require_api_key: bool,
}

impl Default for IpcServerConfig {
    fn default() -> Self {
        Self {
            service_name: "rwkv.infer.openai".to_string(),
            max_request_bytes: 4 * 1024 * 1024,
            max_response_bytes: 4 * 1024 * 1024,
            max_inflight_requests: 128,
            require_api_key: true,
        }
    }
}

impl IpcServerConfig {
    pub fn validate(&self) -> IpcResult<()> {
        if self.service_name.trim().is_empty() {
            return Err(IpcError::bad_request("ipc service_name cannot be empty"));
        }
        if self.max_request_bytes < 1024 {
            return Err(IpcError::bad_request(
                "ipc max_request_bytes must be >= 1024",
            ));
        }
        if self.max_response_bytes < 1024 {
            return Err(IpcError::bad_request(
                "ipc max_response_bytes must be >= 1024",
            ));
        }
        if self.max_inflight_requests == 0 {
            return Err(IpcError::bad_request(
                "ipc max_inflight_requests must be >= 1",
            ));
        }
        Ok(())
    }
}

pub struct IpcServer {
    config: IpcServerConfig,
    queues: SharedQueueMap,
    gpu_metrics: Arc<GpuMetricsCache>,
    reload_lock: Option<Arc<tokio::sync::Mutex<()>>>,
    infer_cfg_path: Option<PathBuf>,
    build_queues: Option<QueueMapBuilder>,
    auth_cfg: AuthConfig,
    runtime_handle: Handle,
}

impl IpcServer {
    pub fn new(
        config: IpcServerConfig,
        auth_cfg: AuthConfig,
        queues: SharedQueueMap,
        gpu_metrics: Arc<GpuMetricsCache>,
        reload_lock: Option<Arc<tokio::sync::Mutex<()>>>,
        infer_cfg_path: Option<PathBuf>,
        build_queues: Option<QueueMapBuilder>,
    ) -> IpcResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            queues,
            gpu_metrics,
            reload_lock,
            infer_cfg_path,
            build_queues,
            auth_cfg,
            runtime_handle: Handle::try_current().map_err(|e| {
                IpcError::internal(format!("failed to access tokio runtime handle: {e}"))
            })?,
        })
    }

    pub fn spawn(self) -> IpcResult<thread::JoinHandle<IpcResult<()>>> {
        let config = self.config.clone();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel::<IpcResult<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("rwkv-infer-ipc-{}", config.service_name))
            .spawn(move || self.run_loop(started_tx))
            .map_err(|e| IpcError::internal(format!("failed to spawn ipc server thread: {e}")))?;

        match started_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(handle),
            Ok(Err(err)) => {
                let _ = handle.join();
                Err(err)
            }
            Err(err) => {
                let _ = handle.join();
                Err(IpcError::internal(format!(
                    "ipc server startup timed out: {err}"
                )))
            }
        }
    }

    fn run_loop(self, started_tx: std::sync::mpsc::SyncSender<IpcResult<()>>) -> IpcResult<()> {
        let service_name = self
            .config
            .service_name
            .as_str()
            .try_into()
            .map_err(|e| IpcError::bad_request(format!("invalid ipc service name: {e}")))?;

        let node = NodeBuilder::new()
            .create::<ipc::Service>()
            .map_err(|e| IpcError::internal(format!("failed to create ipc node: {e}")))?;

        let service = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .open_or_create()
            .map_err(|e| IpcError::internal(format!("failed to create/open ipc service: {e}")))?;

        let server = service
            .server_builder()
            .initial_max_slice_len(self.config.max_response_bytes)
            .create()
            .map_err(|e| IpcError::internal(format!("failed to create ipc server: {e}")))?;

        let _ = started_tx.send(Ok(()));

        loop {
            match server.receive() {
                Ok(Some(active_request)) => {
                    if let Err(err) = self.handle_active_request(&active_request) {
                        log::error!("ipc request handling failed: {err}");
                    }
                }
                Ok(None) => {
                    let _ = node.wait(Duration::from_millis(1));
                }
                Err(err) => {
                    return Err(IpcError::internal(format!(
                        "ipc server receive failed: {err}"
                    )));
                }
            }
        }
    }

    fn handle_active_request<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let request = match decode_request(active_request.payload()) {
            Ok(request) => request,
            Err(err) => {
                self.send_frame(
                    active_request,
                    IpcResponse {
                        version: IPC_PROTOCOL_VERSION,
                        request_id: Uuid::nil(),
                        route: RouteId::Handshake,
                        kind: ResponseKind::Error,
                        status_code: err.status_code(),
                        payload: encode_json(&err.openai_error_response())?,
                    },
                )?;
                return Ok(());
            }
        };

        if request.payload.len() > self.config.max_request_bytes {
            return self.send_ipc_error(
                active_request,
                &request,
                IpcError::bad_request(format!(
                    "ipc request payload exceeds limit {}",
                    self.config.max_request_bytes
                )),
            );
        }

        if request.route != RouteId::Handshake && request.version != IPC_PROTOCOL_VERSION {
            return self.send_ipc_error(
                active_request,
                &request,
                IpcError::bad_request(format!(
                    "ipc protocol version mismatch: request={}, server={}",
                    request.version, IPC_PROTOCOL_VERSION
                )),
            );
        }

        if self.config.require_api_key && route_requires_auth(request.route) {
            if let Err(err) = check_api_key(request.api_key.as_deref(), &self.auth_cfg) {
                return self.send_ipc_error(
                    active_request,
                    &request,
                    IpcError::unauthorized(err.to_string()),
                );
            }
        }

        self.dispatch_request(active_request, request)
    }

    fn dispatch_request<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        match request.route {
            RouteId::Handshake => self.handle_handshake(active_request, request),
            RouteId::Health => {
                let queues = self
                    .queues
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let payload = encode_json(&crate::services::health::health(
                    &queues,
                    self.gpu_metrics.as_ref(),
                ))?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            RouteId::ModelsList => {
                let queues = self
                    .queues
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let payload = encode_json(&crate::services::models::models(&queues))?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            RouteId::Completions => self.handle_completions(active_request, request),
            RouteId::ChatCompletions => self.handle_chat_completions(active_request, request),
            RouteId::AdminModelsReload => self.handle_models_reload(active_request, request),
            RouteId::ResponsesCreate
            | RouteId::ResponsesGet
            | RouteId::ResponsesDelete
            | RouteId::ResponsesCancel
            | RouteId::Embeddings
            | RouteId::ImagesGenerations
            | RouteId::AudioSpeech => self.send_ipc_error(
                active_request,
                &request,
                IpcError::not_supported(format!(
                    "route {:?} is not migrated to the new infer implementation yet",
                    request.route
                )),
            ),
        }
    }

    fn handle_handshake<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let handshake = decode_json::<HandshakeRequest>(&request.payload, "handshake payload")?;
        let accepted =
            request.version == IPC_PROTOCOL_VERSION && handshake.version == IPC_PROTOCOL_VERSION;
        let response = HandshakeResponse {
            accepted,
            server_version: IPC_PROTOCOL_VERSION,
            reason: if accepted {
                None
            } else {
                Some(format!(
                    "protocol mismatch: request={}, payload={}, server={}",
                    request.version, handshake.version, IPC_PROTOCOL_VERSION
                ))
            },
        };

        let status = if accepted { 200 } else { 400 };
        let payload = encode_json(&response)?;
        self.send_data_then_done(active_request, &request, payload, status)
    }

    fn handle_completions<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let mut completion = decode_json::<CompletionsReq>(&request.payload, "completion request")?;
        completion.stream = Some(request.stream);
        let handle = {
            let queues = self
                .queues
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            select_model_queue(&queues, &completion.model).map_err(IpcError::from)
        };
        let handle = match handle {
            Ok(handle) => handle,
            Err(err) => return self.send_ipc_error(active_request, &request, err),
        };

        let run = self
            .runtime_handle
            .block_on(crate::services::completions::completions(
                handle, completion,
            ))
            .map_err(IpcError::from);

        let run = match run {
            Ok(run) => run,
            Err(err) => return self.send_ipc_error(active_request, &request, err),
        };

        if run.stream_requested {
            let mut run = run;
            let mut text_offset = 0usize;
            let mut finish_meta = None;

            while let Some(event) = self.runtime_handle.block_on(async { run.rx.recv().await }) {
                match event {
                    crate::cores::queue::QueueEvent::Delta(delta) => {
                        let chunk = run.stream_chunk(&delta, text_offset);
                        text_offset += delta.text.chars().count();
                        self.send_data(active_request, &request, encode_json(&chunk)?, 200)?;
                    }
                    crate::cores::queue::QueueEvent::Done(meta) => {
                        finish_meta = Some(meta);
                        break;
                    }
                }
            }

            if let Some(finish_meta) = finish_meta {
                let final_chunk = run.finish_chunk(&finish_meta);
                self.send_data(active_request, &request, encode_json(&final_chunk)?, 200)?;
            }
            return self.send_done(active_request, &request);
        }

        let resp: CompletionsResp = self.runtime_handle.block_on(run.collect());
        self.send_data_then_done(active_request, &request, encode_json(&resp)?, 200)
    }

    fn handle_chat_completions<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let mut chat =
            decode_json::<ChatCompletionsReq>(&request.payload, "chat completion request")?;
        chat.stream = Some(request.stream);
        let handle = {
            let queues = self
                .queues
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            select_model_queue(&queues, &chat.model).map_err(IpcError::from)
        };
        let handle = match handle {
            Ok(handle) => handle,
            Err(err) => return self.send_ipc_error(active_request, &request, err),
        };

        let run = self
            .runtime_handle
            .block_on(crate::services::chat::completions::chat_completions(
                handle, chat,
            ))
            .map_err(IpcError::from);

        let run = match run {
            Ok(run) => run,
            Err(err) => return self.send_ipc_error(active_request, &request, err),
        };

        if run.stream_requested {
            let mut run = run;
            let mut stream_state = run.new_stream_state();
            let role_chunk = run.stream_role_chunk();
            self.send_data(active_request, &request, encode_json(&role_chunk)?, 200)?;

            let mut finish_meta = None;
            while let Some(event) = self.runtime_handle.block_on(async { run.rx.recv().await }) {
                match event {
                    crate::cores::queue::QueueEvent::Delta(delta) => {
                        for chunk in run.stream_chunks(&mut stream_state, &delta) {
                            self.send_data(active_request, &request, encode_json(&chunk)?, 200)?;
                        }
                    }
                    crate::cores::queue::QueueEvent::Done(meta) => {
                        finish_meta = Some(meta);
                        break;
                    }
                }
            }

            if let Some(finish_meta) = finish_meta {
                for chunk in run.finish_chunks(&mut stream_state, &finish_meta) {
                    self.send_data(active_request, &request, encode_json(&chunk)?, 200)?;
                }
            }
            return self.send_done(active_request, &request);
        }

        let resp: ChatCompletionResp = self.runtime_handle.block_on(run.collect());
        self.send_data_then_done(active_request, &request, encode_json(&resp)?, 200)
    }

    fn handle_models_reload<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let req = decode_json::<ModelsReloadReq>(&request.payload, "reload request")?;
        let resp = self
            .runtime_handle
            .block_on(crate::services::admin::models::reload::reload_models(
                &self.queues,
                self.reload_lock.as_ref(),
                self.infer_cfg_path.as_deref(),
                self.build_queues.as_ref(),
                req,
            ))
            .map_err(IpcError::from);

        let resp: ModelsReloadResp = match resp {
            Ok(resp) => resp,
            Err(err) => return self.send_ipc_error(active_request, &request, err),
        };

        self.send_data_then_done(active_request, &request, encode_json(&resp)?, 200)
    }

    fn send_data_then_done<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
        payload: Vec<u8>,
        status_code: u16,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        self.send_data(active_request, request, payload, status_code)?;
        self.send_done(active_request, request)
    }

    fn send_data<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
        payload: Vec<u8>,
        status_code: u16,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        self.send_frame(
            active_request,
            IpcResponse {
                version: IPC_PROTOCOL_VERSION,
                request_id: request.request_id,
                route: request.route,
                kind: ResponseKind::Data,
                status_code,
                payload,
            },
        )
    }

    fn send_done<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        self.send_frame(
            active_request,
            IpcResponse {
                version: IPC_PROTOCOL_VERSION,
                request_id: request.request_id,
                route: request.route,
                kind: ResponseKind::Done,
                status_code: 200,
                payload: Vec::new(),
            },
        )
    }

    fn send_ipc_error<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
        error: IpcError,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        self.send_frame(
            active_request,
            IpcResponse {
                version: IPC_PROTOCOL_VERSION,
                request_id: request.request_id,
                route: request.route,
                kind: ResponseKind::Error,
                status_code: error.status_code(),
                payload: encode_json(&error.openai_error_response())?,
            },
        )
    }

    fn send_frame<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        response: IpcResponse,
    ) -> IpcResult<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        if response.payload.len() > self.config.max_response_bytes {
            return Err(IpcError::bad_request(format!(
                "ipc response payload exceeds limit {}",
                self.config.max_response_bytes
            )));
        }

        let bytes = encode_response(&response)?;
        let mut response_uninit = active_request
            .loan_slice_uninit(bytes.len())
            .map_err(|e| IpcError::internal(format!("failed to loan ipc response: {e}")))?;
        for (slot, value) in response_uninit.payload_mut().iter_mut().zip(bytes.iter()) {
            slot.write(*value);
        }
        let response = unsafe { response_uninit.assume_init() };
        response
            .send()
            .map_err(|e| IpcError::internal(format!("failed to send ipc response: {e}")))
    }
}

fn route_requires_auth(route: RouteId) -> bool {
    matches!(
        route,
        RouteId::Completions
            | RouteId::ChatCompletions
            | RouteId::ResponsesCreate
            | RouteId::ResponsesGet
            | RouteId::ResponsesDelete
            | RouteId::ResponsesCancel
            | RouteId::Embeddings
            | RouteId::ImagesGenerations
            | RouteId::AudioSpeech
            | RouteId::AdminModelsReload
    )
}
