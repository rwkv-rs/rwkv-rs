use core::fmt::Debug;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use iceoryx2::active_request::ActiveRequest;
use iceoryx2::prelude::{NodeBuilder, ZeroCopySend, ipc};
use tokio::runtime::Handle;

use crate::api::ApiService;
use crate::auth::{AuthConfig, check_api_key_token};
use crate::service::RuntimeManager;

use super::protocol::{
    HandshakeRequest, HandshakeResponse, IPC_PROTOCOL_VERSION, IpcRequest, IpcResponse,
    ResponseKind, RouteId, decode_json, decode_request, encode_json, encode_response,
};

#[derive(Clone, Debug)]
pub struct IpcServerConfig {
    pub service_name: String,
    pub max_request_bytes: usize,
    pub max_response_bytes: usize,
    pub max_inflight_requests: usize,
    pub require_api_key: bool,
}

impl IpcServerConfig {
    pub fn validate(&self) -> crate::Result<()> {
        if self.service_name.trim().is_empty() {
            return Err(crate::Error::bad_request(
                "ipc service_name cannot be empty",
            ));
        }
        if self.max_request_bytes < 1024 {
            return Err(crate::Error::bad_request(
                "ipc max_request_bytes must be >= 1024",
            ));
        }
        if self.max_response_bytes < 1024 {
            return Err(crate::Error::bad_request(
                "ipc max_response_bytes must be >= 1024",
            ));
        }
        if self.max_inflight_requests == 0 {
            return Err(crate::Error::bad_request(
                "ipc max_inflight_requests must be >= 1",
            ));
        }
        Ok(())
    }
}

pub struct IpcServer {
    config: IpcServerConfig,
    api: ApiService,
    auth_cfg: AuthConfig,
    runtime_handle: Handle,
}

impl IpcServer {
    pub fn from_runtime_manager(
        runtime_manager: Arc<RuntimeManager>,
        auth_cfg: AuthConfig,
    ) -> crate::Result<Self> {
        let config = IpcServerConfig {
            service_name: runtime_manager.ipc_service_name(),
            max_request_bytes: runtime_manager.ipc_max_request_bytes(),
            max_response_bytes: runtime_manager.ipc_max_response_bytes(),
            max_inflight_requests: runtime_manager.ipc_max_inflight_requests(),
            require_api_key: runtime_manager.ipc_require_api_key(),
        };
        config.validate()?;

        Ok(Self {
            config,
            api: ApiService::new(runtime_manager),
            auth_cfg,
            runtime_handle: Handle::try_current().map_err(|e| {
                crate::Error::internal_with_source("failed to access tokio runtime handle", e)
            })?,
        })
    }

    pub fn spawn(self) -> crate::Result<thread::JoinHandle<crate::Result<()>>> {
        let config = self.config.clone();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel::<crate::Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("rwkv-infer-ipc-{}", config.service_name))
            .spawn(move || self.run_loop(started_tx))
            .map_err(|e| {
                crate::Error::internal_with_source("failed to spawn ipc server thread", e)
            })?;

        match started_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(handle),
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            Err(e) => {
                let _ = handle.join();
                Err(crate::Error::internal_with_source(
                    "ipc server startup timed out",
                    e,
                ))
            }
        }
    }

    fn run_loop(
        self,
        started_tx: std::sync::mpsc::SyncSender<crate::Result<()>>,
    ) -> crate::Result<()> {
        let service_name =
            self.config.service_name.as_str().try_into().map_err(|e| {
                crate::Error::bad_request_with_source("invalid ipc service name", e)
            })?;

        let node = NodeBuilder::new()
            .create::<ipc::Service>()
            .map_err(|e| crate::Error::internal_with_source("failed to create ipc node", e))?;

        let service = node
            .service_builder(&service_name)
            .request_response::<[u8], [u8]>()
            .open_or_create()
            .map_err(|e| {
                crate::Error::internal_with_source("failed to create/open ipc service", e)
            })?;

        let server = service
            .server_builder()
            .initial_max_slice_len(self.config.max_response_bytes)
            .create()
            .map_err(|e| crate::Error::internal_with_source("failed to create ipc server", e))?;

        let _ = started_tx.send(Ok(()));

        loop {
            match server.receive() {
                Ok(Some(active_request)) => {
                    if let Err(e) = self.handle_active_request(&active_request) {
                        log::error!("ipc request handling failed: {}", e.format_chain());
                    }
                }
                Ok(None) => {
                    let _ = node.wait(Duration::from_millis(1));
                }
                Err(e) => {
                    return Err(crate::Error::internal_with_source(
                        "ipc server receive failed",
                        e,
                    ));
                }
            }
        }
    }

    fn handle_active_request<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let request = match decode_request(active_request.payload()) {
            Ok(request) => request,
            Err(e) => {
                let payload =
                    encode_json(&crate::server::OpenAiErrorResponse::from_infer_error(&e))?;
                self.send_frame(
                    active_request,
                    IpcResponse {
                        version: IPC_PROTOCOL_VERSION,
                        request_id: uuid::Uuid::nil(),
                        route: RouteId::Handshake,
                        kind: ResponseKind::Error,
                        status_code: 400,
                        payload,
                    },
                )?;
                return Ok(());
            }
        };

        if request.payload.len() > self.config.max_request_bytes {
            return self.send_infer_error(
                active_request,
                &request,
                crate::Error::bad_request(format!(
                    "ipc request payload exceeds limit {}",
                    self.config.max_request_bytes
                )),
                None,
            );
        }

        if request.route != RouteId::Handshake && request.version != IPC_PROTOCOL_VERSION {
            return self.send_infer_error(
                active_request,
                &request,
                crate::Error::bad_request(format!(
                    "ipc protocol version mismatch: request={}, server={}",
                    request.version, IPC_PROTOCOL_VERSION
                )),
                None,
            );
        }

        if self.config.require_api_key && route_requires_auth(request.route) {
            if let Err(auth_err) = check_api_key_token(request.api_key.as_deref(), &self.auth_cfg) {
                let payload = encode_json(&crate::server::OpenAiErrorResponse::unauthorized(
                    auth_err.message(),
                ))?;
                self.send_frame(
                    active_request,
                    IpcResponse {
                        version: IPC_PROTOCOL_VERSION,
                        request_id: request.request_id,
                        route: request.route,
                        kind: ResponseKind::Error,
                        status_code: 401,
                        payload,
                    },
                )?;
                return Ok(());
            }
        }

        self.dispatch_request(active_request, request)
    }

    fn dispatch_request<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        match request.route {
            RouteId::Handshake => self.handle_handshake(active_request, request),
            RouteId::Health => {
                let payload = encode_json(&self.api.health())?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            RouteId::ModelsList => {
                let payload = encode_json(&self.api.models())?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            RouteId::Completions => self.handle_completions(active_request, request),
            RouteId::ChatCompletions => self.handle_chat_completions(active_request, request),
            RouteId::ResponsesCreate => self.handle_responses_create(active_request, request),
            RouteId::ResponsesGet => self.handle_responses_get(active_request, request),
            RouteId::ResponsesDelete => self.handle_responses_delete(active_request, request),
            RouteId::ResponsesCancel => self.handle_responses_cancel(active_request, request),
            RouteId::Embeddings => self.handle_not_supported_route(active_request, request),
            RouteId::ImagesGenerations => self.handle_not_supported_route(active_request, request),
            RouteId::AudioSpeech => self.handle_not_supported_route(active_request, request),
            RouteId::AdminModelsReload => self.handle_admin_models_reload(active_request, request),
        }
    }

    fn handle_handshake<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
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
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let mut completion = decode_json::<crate::server::CompletionRequest>(
            &request.payload,
            "completion request",
        )?;
        completion.stream = Some(request.stream);

        let run = match self
            .runtime_handle
            .block_on(self.api.completions(completion))
        {
            Ok(run) => run,
            Err(e) => return self.send_infer_error(active_request, &request, e, None),
        };

        if run.stream_requested {
            let crate::api::CompletionRun {
                id,
                created,
                model,
                stream_requested: _,
                mut rx,
            } = run;

            loop {
                let ev = self.runtime_handle.block_on(async { rx.recv().await });
                match ev {
                    Some(crate::types::EngineEvent::Text(text)) => {
                        let chunk = crate::server::CompletionResponse {
                            id: id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::CompletionResponseChoice {
                                text,
                                index: 0,
                                finish_reason: None,
                            }],
                            timings_ms: None,
                        };
                        let payload = encode_json(&chunk)?;
                        self.send_data(active_request, &request, payload, 200)?;
                    }
                    Some(crate::types::EngineEvent::Done(meta)) => {
                        let final_chunk = crate::server::CompletionResponse {
                            id: id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::CompletionResponseChoice {
                                text: String::new(),
                                index: 0,
                                finish_reason: Some(meta.reason.as_openai_str().to_string()),
                            }],
                            timings_ms: meta.timings_ms.clone(),
                        };
                        let payload = encode_json(&final_chunk)?;
                        self.send_data(active_request, &request, payload, 200)?;
                        self.send_done(active_request, &request)?;
                        break;
                    }
                    Some(crate::types::EngineEvent::Error(msg)) => {
                        return self.send_infer_error(
                            active_request,
                            &request,
                            crate::Error::internal(msg),
                            None,
                        );
                    }
                    None => {
                        self.send_done(active_request, &request)?;
                        break;
                    }
                }
            }

            return Ok(());
        }

        match self.runtime_handle.block_on(run.collect()) {
            Ok(resp) => {
                let payload = encode_json(&resp)?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            Err(e) => self.send_infer_error(active_request, &request, e, None),
        }
    }

    fn handle_chat_completions<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let mut chat = decode_json::<crate::server::ChatCompletionRequest>(
            &request.payload,
            "chat completion request",
        )?;
        chat.stream = Some(request.stream);

        let run = match self
            .runtime_handle
            .block_on(self.api.chat_completions(chat))
        {
            Ok(run) => run,
            Err(e) => return self.send_infer_error(active_request, &request, e, None),
        };

        if run.stream_requested {
            let crate::api::ChatCompletionRun {
                id,
                created,
                model,
                stream_requested: _,
                mut rx,
            } = run;

            loop {
                let ev = self.runtime_handle.block_on(async { rx.recv().await });
                match ev {
                    Some(crate::types::EngineEvent::Text(text)) => {
                        let chunk = crate::server::ChatCompletionResponse {
                            id: id.clone(),
                            object: "chat.completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::ChatCompletionResponseChoice {
                                index: 0,
                                message: crate::server::ChatMessage {
                                    role: "assistant".to_string(),
                                    content: text,
                                },
                                finish_reason: None,
                            }],
                            timings_ms: None,
                        };
                        let payload = encode_json(&chunk)?;
                        self.send_data(active_request, &request, payload, 200)?;
                    }
                    Some(crate::types::EngineEvent::Done(meta)) => {
                        let final_chunk = crate::server::ChatCompletionResponse {
                            id: id.clone(),
                            object: "chat.completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![crate::server::ChatCompletionResponseChoice {
                                index: 0,
                                message: crate::server::ChatMessage {
                                    role: "assistant".to_string(),
                                    content: String::new(),
                                },
                                finish_reason: Some(meta.reason.as_openai_str().to_string()),
                            }],
                            timings_ms: meta.timings_ms.clone(),
                        };
                        let payload = encode_json(&final_chunk)?;
                        self.send_data(active_request, &request, payload, 200)?;
                        self.send_done(active_request, &request)?;
                        break;
                    }
                    Some(crate::types::EngineEvent::Error(msg)) => {
                        return self.send_infer_error(
                            active_request,
                            &request,
                            crate::Error::internal(msg),
                            None,
                        );
                    }
                    None => {
                        self.send_done(active_request, &request)?;
                        break;
                    }
                }
            }

            return Ok(());
        }

        match self.runtime_handle.block_on(run.collect()) {
            Ok(resp) => {
                let payload = encode_json(&resp)?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            Err(e) => self.send_infer_error(active_request, &request, e, None),
        }
    }

    fn handle_responses_create<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let req = decode_json::<crate::server::ResponsesCreateRequest>(
            &request.payload,
            "responses create request",
        )?;
        match self.runtime_handle.block_on(self.api.responses_create(req)) {
            Ok(resp) => {
                let payload = encode_json(&resp)?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            Err(e) => self.send_infer_error(active_request, &request, e, None),
        }
    }

    fn handle_responses_get<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let req =
            decode_json::<crate::server::ResponseIdRequest>(&request.payload, "responses id")?;
        match self.api.responses_get(req) {
            Some(resp) => {
                let payload = encode_json(&resp)?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            None => self.send_openai_error(
                active_request,
                &request,
                404,
                crate::server::OpenAiErrorResponse::bad_request("response not found"),
            ),
        }
    }

    fn handle_responses_delete<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let req =
            decode_json::<crate::server::ResponseIdRequest>(&request.payload, "responses id")?;
        match self.api.responses_delete(req) {
            Some(resp) => {
                let payload = encode_json(&resp)?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            None => self.send_openai_error(
                active_request,
                &request,
                404,
                crate::server::OpenAiErrorResponse::bad_request("response not found"),
            ),
        }
    }

    fn handle_responses_cancel<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let req =
            decode_json::<crate::server::ResponseIdRequest>(&request.payload, "responses id")?;
        match self.api.responses_cancel(req) {
            Ok(()) => self.send_data_then_done(
                active_request,
                &request,
                encode_json(&sonic_rs::json!({"ok": true}))?,
                200,
            ),
            Err(e) => self.send_infer_error(active_request, &request, e, Some(501)),
        }
    }

    fn handle_not_supported_route<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let result = match request.route {
            RouteId::Embeddings => self.api.embeddings(),
            RouteId::ImagesGenerations => self.api.images_generations(),
            RouteId::AudioSpeech => self.api.audio_speech(),
            _ => Err(crate::Error::not_supported("unsupported route")),
        };
        match result {
            Ok(()) => self.send_data_then_done(
                active_request,
                &request,
                encode_json(&sonic_rs::json!({}))?,
                200,
            ),
            Err(e) => self.send_infer_error(active_request, &request, e, Some(501)),
        }
    }

    fn handle_admin_models_reload<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: IpcRequest,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let req =
            decode_json::<crate::server::ReloadModelsRequest>(&request.payload, "reload request")?;
        match self
            .runtime_handle
            .block_on(self.api.admin_models_reload(req))
        {
            Ok(resp) => {
                let payload = encode_json(&resp)?;
                self.send_data_then_done(active_request, &request, payload, 200)
            }
            Err(e) => self.send_infer_error(active_request, &request, e, None),
        }
    }

    fn send_data_then_done<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
        payload: Vec<u8>,
        status_code: u16,
    ) -> crate::Result<()>
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
    ) -> crate::Result<()>
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
    ) -> crate::Result<()>
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

    fn send_openai_error<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
        status_code: u16,
        error: crate::server::OpenAiErrorResponse,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let payload = encode_json(&error)?;
        self.send_frame(
            active_request,
            IpcResponse {
                version: IPC_PROTOCOL_VERSION,
                request_id: request.request_id,
                route: request.route,
                kind: ResponseKind::Error,
                status_code,
                payload,
            },
        )
    }

    fn send_infer_error<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        request: &IpcRequest,
        error: crate::Error,
        status_override: Option<u16>,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        let status =
            status_override.unwrap_or_else(|| if error.is_client_error() { 400 } else { 500 });
        self.send_openai_error(
            active_request,
            request,
            status,
            crate::server::OpenAiErrorResponse::from_infer_error(&error),
        )
    }

    fn send_frame<RequestHeader, ResponseHeader>(
        &self,
        active_request: &ActiveRequest<ipc::Service, [u8], RequestHeader, [u8], ResponseHeader>,
        response: IpcResponse,
    ) -> crate::Result<()>
    where
        RequestHeader: Debug + ZeroCopySend,
        ResponseHeader: Debug + Default + ZeroCopySend,
    {
        if response.payload.len() > self.config.max_response_bytes {
            return Err(crate::Error::bad_request(format!(
                "ipc response payload exceeds limit {}",
                self.config.max_response_bytes
            )));
        }

        let bytes = encode_response(&response)?;
        let mut response_uninit = active_request
            .loan_slice_uninit(bytes.len())
            .map_err(|e| crate::Error::internal_with_source("failed to loan ipc response", e))?;
        for (slot, value) in response_uninit.payload_mut().iter_mut().zip(bytes.iter()) {
            slot.write(*value);
        }
        let response = unsafe { response_uninit.assume_init() };
        response
            .send()
            .map_err(|e| crate::Error::internal_with_source("failed to send ipc response", e))
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
