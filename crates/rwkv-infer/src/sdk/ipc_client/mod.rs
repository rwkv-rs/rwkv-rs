#![cfg(feature = "ipc")]

use std::time::{Duration, Instant};

use iceoryx2::{
    port::client::Client,
    prelude::{NodeBuilder, ipc},
};
use uuid::Uuid;

use crate::{
    dtos::{
        admin::models::reload::{ModelsReloadReq, ModelsReloadResp},
        chat::completions::{ChatCompletionResp, ChatCompletionsChunkResponse, ChatCompletionsReq},
        completions::{CompletionsReq, CompletionsResp},
        health::HealthResp,
        models::ModelsResp,
    },
    routes::ipc_api::protocol::{
        HandshakeRequest,
        HandshakeResponse,
        IPC_PROTOCOL_VERSION,
        IpcError,
        IpcResponse,
        IpcResult,
        ResponseKind,
        RouteId,
        decode_json,
        decode_response,
        encode_json,
        encode_request,
    },
};

#[derive(Clone, Debug)]
pub struct IpcClientConfig {
    pub service_name: String,
    pub api_key: Option<String>,
    pub max_request_bytes: usize,
    pub max_response_bytes: usize,
    pub request_timeout: Duration,
}

impl Default for IpcClientConfig {
    fn default() -> Self {
        Self {
            service_name: "rwkv.infer.openai".to_string(),
            api_key: None,
            max_request_bytes: 4 * 1024 * 1024,
            max_response_bytes: 4 * 1024 * 1024,
            request_timeout: Duration::from_secs(30),
        }
    }
}

impl IpcClientConfig {
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
        if self.request_timeout.is_zero() {
            return Err(IpcError::bad_request(
                "ipc request_timeout must be greater than 0",
            ));
        }
        Ok(())
    }
}

pub struct IpcOpenAiClient {
    port: Client<ipc::Service, [u8], (), [u8], ()>,
    api_key: Option<String>,
    max_request_bytes: usize,
    max_response_bytes: usize,
    request_timeout: Duration,
}

impl IpcOpenAiClient {
    pub fn connect(config: IpcClientConfig) -> IpcResult<Self> {
        config.validate()?;

        let service_name = config
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
            .map_err(|e| IpcError::internal(format!("failed to open/create ipc service: {e}")))?;
        let port = service
            .client_builder()
            .initial_max_slice_len(config.max_request_bytes)
            .create()
            .map_err(|e| IpcError::internal(format!("failed to create ipc client: {e}")))?;

        let client = Self {
            port,
            api_key: config.api_key,
            max_request_bytes: config.max_request_bytes,
            max_response_bytes: config.max_response_bytes,
            request_timeout: config.request_timeout,
        };

        client.handshake()?;
        Ok(client)
    }

    pub fn health(&self) -> IpcResult<HealthResp> {
        let responses = self.send_request(RouteId::Health, false, Vec::new())?;
        parse_single_data(responses, "health response")
    }

    pub fn models(&self) -> IpcResult<ModelsResp> {
        let responses = self.send_request(RouteId::ModelsList, false, Vec::new())?;
        parse_single_data(responses, "models response")
    }

    pub fn completions(&self, mut request: CompletionsReq) -> IpcResult<CompletionsResp> {
        request.stream = Some(false);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::Completions, false, payload)?;
        parse_single_data(responses, "completion response")
    }

    pub fn completions_stream(
        &self,
        mut request: CompletionsReq,
    ) -> IpcResult<Vec<CompletionsResp>> {
        request.stream = Some(true);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::Completions, true, payload)?;
        parse_stream_data(responses, "completion stream")
    }

    pub fn chat_completions(
        &self,
        mut request: ChatCompletionsReq,
    ) -> IpcResult<ChatCompletionResp> {
        request.stream = Some(false);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::ChatCompletions, false, payload)?;
        parse_single_data(responses, "chat completion response")
    }

    pub fn chat_completions_stream(
        &self,
        mut request: ChatCompletionsReq,
    ) -> IpcResult<Vec<ChatCompletionsChunkResponse>> {
        request.stream = Some(true);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::ChatCompletions, true, payload)?;
        parse_stream_data(responses, "chat completion stream")
    }

    pub fn reload_models(&self, request: ModelsReloadReq) -> IpcResult<ModelsReloadResp> {
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::AdminModelsReload, false, payload)?;
        parse_single_data(responses, "admin models reload")
    }

    fn handshake(&self) -> IpcResult<()> {
        let payload = encode_json(&HandshakeRequest {
            version: IPC_PROTOCOL_VERSION,
            client_name: Some("rwkv-infer-ipc-sdk".to_string()),
        })?;
        let responses = self.send_request(RouteId::Handshake, false, payload)?;
        let handshake: HandshakeResponse = parse_single_data(responses, "handshake response")?;
        if !handshake.accepted {
            return Err(IpcError::bad_request(
                handshake
                    .reason
                    .unwrap_or_else(|| "server rejected handshake".to_string()),
            ));
        }
        if handshake.server_version != IPC_PROTOCOL_VERSION {
            return Err(IpcError::bad_request(format!(
                "ipc protocol mismatch: server={}, client={}",
                handshake.server_version, IPC_PROTOCOL_VERSION
            )));
        }
        Ok(())
    }

    fn send_request(
        &self,
        route: RouteId,
        stream: bool,
        payload: Vec<u8>,
    ) -> IpcResult<Vec<IpcResponse>> {
        if payload.len() > self.max_request_bytes {
            return Err(IpcError::bad_request(format!(
                "ipc request payload exceeds limit {}",
                self.max_request_bytes
            )));
        }

        let request_frame = crate::routes::ipc_api::IpcRequest {
            version: IPC_PROTOCOL_VERSION,
            request_id: Uuid::new_v4(),
            route,
            stream,
            api_key: self.api_key.clone(),
            payload,
        };
        let encoded = encode_request(&request_frame)?;
        if encoded.len() > self.max_request_bytes {
            return Err(IpcError::bad_request(format!(
                "ipc request frame exceeds limit {}",
                self.max_request_bytes
            )));
        }

        let mut request_uninit = self
            .port
            .loan_slice_uninit(encoded.len())
            .map_err(|e| IpcError::internal(format!("failed to loan ipc request: {e}")))?;
        for (slot, value) in request_uninit.payload_mut().iter_mut().zip(encoded.iter()) {
            slot.write(*value);
        }
        let request = unsafe { request_uninit.assume_init() };
        let pending = request
            .send()
            .map_err(|e| IpcError::internal(format!("failed to send ipc request: {e}")))?;

        let start = Instant::now();
        let mut responses = Vec::new();
        loop {
            match pending.receive() {
                Ok(Some(sample)) => {
                    let frame = decode_response(sample.payload())?;
                    if frame.request_id != request_frame.request_id || frame.route != route {
                        continue;
                    }
                    if frame.payload.len() > self.max_response_bytes {
                        return Err(IpcError::bad_request(format!(
                            "ipc response payload exceeds limit {}",
                            self.max_response_bytes
                        )));
                    }
                    let is_terminal =
                        matches!(frame.kind, ResponseKind::Done | ResponseKind::Error);
                    responses.push(frame);
                    if is_terminal {
                        break;
                    }
                }
                Ok(None) => {
                    if start.elapsed() >= self.request_timeout {
                        return Err(IpcError::internal(format!(
                            "ipc request timed out after {:?}",
                            self.request_timeout
                        )));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    return Err(IpcError::internal(format!(
                        "failed to receive ipc response: {e}"
                    )));
                }
            }
        }

        Ok(responses)
    }
}

fn parse_single_data<T: serde::de::DeserializeOwned>(
    responses: Vec<IpcResponse>,
    context: &str,
) -> IpcResult<T> {
    let mut last_data = None;
    for response in responses {
        match response.kind {
            ResponseKind::Data => {
                last_data = Some(response.payload);
            }
            ResponseKind::Done => {}
            ResponseKind::Error => return Err(frame_to_error(response)),
        }
    }
    let payload = last_data
        .ok_or_else(|| IpcError::internal(format!("missing data frame for {context} response")))?;
    decode_json::<T>(&payload, context)
}

fn parse_stream_data<T: serde::de::DeserializeOwned>(
    responses: Vec<IpcResponse>,
    context: &str,
) -> IpcResult<Vec<T>> {
    let mut out = Vec::new();
    for response in responses {
        match response.kind {
            ResponseKind::Data => out.push(decode_json::<T>(&response.payload, context)?),
            ResponseKind::Done => {}
            ResponseKind::Error => return Err(frame_to_error(response)),
        }
    }
    Ok(out)
}

fn frame_to_error(frame: IpcResponse) -> IpcError {
    match decode_json::<crate::dtos::errors::OpenAiErrorResponse>(&frame.payload, "error frame") {
        Ok(err) => IpcError::from_status_and_body(frame.status_code, err),
        Err(parse_err) => {
            IpcError::internal(format!("failed to decode ipc error frame: {parse_err}"))
        }
    }
}
