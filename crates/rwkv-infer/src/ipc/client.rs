use std::time::{Duration, Instant};

use iceoryx2::port::client::Client;
use iceoryx2::prelude::{NodeBuilder, ipc};
use uuid::Uuid;

use super::protocol::{
    HandshakeRequest, HandshakeResponse, IPC_PROTOCOL_VERSION, IpcRequest, IpcResponse,
    ResponseKind, RouteId, decode_json, decode_response, encode_json, encode_request,
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
        if self.request_timeout.is_zero() {
            return Err(crate::Error::bad_request(
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
    pub fn connect(config: IpcClientConfig) -> crate::Result<Self> {
        config.validate()?;

        let service_name =
            config.service_name.as_str().try_into().map_err(|e| {
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
                crate::Error::internal_with_source("failed to open/create ipc service", e)
            })?;
        let port = service
            .client_builder()
            .initial_max_slice_len(config.max_request_bytes)
            .create()
            .map_err(|e| crate::Error::internal_with_source("failed to create ipc client", e))?;

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

    pub fn health(&self) -> crate::Result<crate::server::HealthResponse> {
        let responses = self.send_request(RouteId::Health, false, Vec::new())?;
        parse_single_data(responses, "health response")
    }

    pub fn models(&self) -> crate::Result<crate::server::ModelListResponse> {
        let responses = self.send_request(RouteId::ModelsList, false, Vec::new())?;
        parse_single_data(responses, "models response")
    }

    pub fn completions(
        &self,
        mut request: crate::server::CompletionRequest,
    ) -> crate::Result<crate::server::CompletionResponse> {
        request.stream = Some(false);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::Completions, false, payload)?;
        parse_single_data(responses, "completion response")
    }

    pub fn completions_stream(
        &self,
        mut request: crate::server::CompletionRequest,
    ) -> crate::Result<Vec<crate::server::CompletionResponse>> {
        request.stream = Some(true);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::Completions, true, payload)?;
        parse_stream_data(responses, "completion stream")
    }

    pub fn chat_completions(
        &self,
        mut request: crate::server::ChatCompletionRequest,
    ) -> crate::Result<crate::server::ChatCompletionResponse> {
        request.stream = Some(false);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::ChatCompletions, false, payload)?;
        parse_single_data(responses, "chat completion response")
    }

    pub fn chat_completions_stream(
        &self,
        mut request: crate::server::ChatCompletionRequest,
    ) -> crate::Result<Vec<crate::server::ChatCompletionResponse>> {
        request.stream = Some(true);
        let payload = encode_json(&request)?;
        let responses = self.send_request(RouteId::ChatCompletions, true, payload)?;
        parse_stream_data(responses, "chat completion stream")
    }

    pub fn responses_create(
        &self,
        request: crate::server::ResponsesCreateRequest,
    ) -> crate::Result<crate::server::ResponsesResource> {
        let responses =
            self.send_request(RouteId::ResponsesCreate, false, encode_json(&request)?)?;
        parse_single_data(responses, "responses create")
    }

    pub fn responses_get(
        &self,
        request: crate::server::ResponseIdRequest,
    ) -> crate::Result<crate::server::ResponsesResource> {
        let responses = self.send_request(RouteId::ResponsesGet, false, encode_json(&request)?)?;
        parse_single_data(responses, "responses get")
    }

    pub fn responses_delete(
        &self,
        request: crate::server::ResponseIdRequest,
    ) -> crate::Result<crate::server::DeleteResponse> {
        let responses =
            self.send_request(RouteId::ResponsesDelete, false, encode_json(&request)?)?;
        parse_single_data(responses, "responses delete")
    }

    pub fn responses_cancel(&self, request: crate::server::ResponseIdRequest) -> crate::Result<()> {
        let responses =
            self.send_request(RouteId::ResponsesCancel, false, encode_json(&request)?)?;
        let _: sonic_rs::Value = parse_single_data(responses, "responses cancel")?;
        Ok(())
    }

    pub fn embeddings(&self) -> crate::Result<()> {
        let responses = self.send_request(RouteId::Embeddings, false, Vec::new())?;
        let _: sonic_rs::Value = parse_single_data(responses, "embeddings")?;
        Ok(())
    }

    pub fn images_generations(&self) -> crate::Result<()> {
        let responses = self.send_request(RouteId::ImagesGenerations, false, Vec::new())?;
        let _: sonic_rs::Value = parse_single_data(responses, "images")?;
        Ok(())
    }

    pub fn audio_speech(&self) -> crate::Result<()> {
        let responses = self.send_request(RouteId::AudioSpeech, false, Vec::new())?;
        let _: sonic_rs::Value = parse_single_data(responses, "audio")?;
        Ok(())
    }

    pub fn admin_models_reload(
        &self,
        request: crate::server::ReloadModelsRequest,
    ) -> crate::Result<crate::server::ReloadModelsResponse> {
        let responses =
            self.send_request(RouteId::AdminModelsReload, false, encode_json(&request)?)?;
        parse_single_data(responses, "admin models reload")
    }

    fn handshake(&self) -> crate::Result<()> {
        let payload = encode_json(&HandshakeRequest {
            version: IPC_PROTOCOL_VERSION,
            client_name: Some("rwkv-infer-ipc-sdk".to_string()),
        })?;
        let responses = self.send_request(RouteId::Handshake, false, payload)?;
        let handshake: HandshakeResponse = parse_single_data(responses, "handshake response")?;
        if !handshake.accepted {
            return Err(crate::Error::bad_request(
                handshake
                    .reason
                    .unwrap_or_else(|| "server rejected handshake".to_string()),
            ));
        }
        if handshake.server_version != IPC_PROTOCOL_VERSION {
            return Err(crate::Error::bad_request(format!(
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
    ) -> crate::Result<Vec<IpcResponse>> {
        if payload.len() > self.max_request_bytes {
            return Err(crate::Error::bad_request(format!(
                "ipc request payload exceeds limit {}",
                self.max_request_bytes
            )));
        }

        let request_frame = IpcRequest {
            version: IPC_PROTOCOL_VERSION,
            request_id: Uuid::new_v4(),
            route,
            stream,
            api_key: self.api_key.clone(),
            payload,
        };
        let encoded = encode_request(&request_frame)?;
        if encoded.len() > self.max_request_bytes {
            return Err(crate::Error::bad_request(format!(
                "ipc request frame exceeds limit {}",
                self.max_request_bytes
            )));
        }

        let mut request_uninit = self
            .port
            .loan_slice_uninit(encoded.len())
            .map_err(|e| crate::Error::internal_with_source("failed to loan ipc request", e))?;
        for (slot, value) in request_uninit.payload_mut().iter_mut().zip(encoded.iter()) {
            slot.write(*value);
        }
        let request = unsafe { request_uninit.assume_init() };
        let pending = request
            .send()
            .map_err(|e| crate::Error::internal_with_source("failed to send ipc request", e))?;

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
                        return Err(crate::Error::bad_request(format!(
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
                        return Err(crate::Error::internal(format!(
                            "ipc request timed out after {:?}",
                            self.request_timeout
                        )));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    return Err(crate::Error::internal_with_source(
                        "failed to receive ipc response",
                        e,
                    ));
                }
            }
        }

        Ok(responses)
    }
}

fn parse_single_data<T: serde::de::DeserializeOwned>(
    responses: Vec<IpcResponse>,
    context: &str,
) -> crate::Result<T> {
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
    let payload = last_data.ok_or_else(|| {
        crate::Error::internal(format!("missing data frame for {context} response"))
    })?;
    decode_json::<T>(&payload, context)
}

fn parse_stream_data<T: serde::de::DeserializeOwned>(
    responses: Vec<IpcResponse>,
    context: &str,
) -> crate::Result<Vec<T>> {
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

fn frame_to_error(frame: IpcResponse) -> crate::Error {
    match decode_json::<crate::server::OpenAiErrorResponse>(&frame.payload, "error frame") {
        Ok(err) => {
            let message = err.error.message;
            if err.error.ty == "not_supported" {
                crate::Error::not_supported(message)
            } else if frame.status_code >= 500 {
                crate::Error::internal(message)
            } else {
                crate::Error::bad_request(message)
            }
        }
        Err(parse_err) => crate::Error::internal(format!(
            "failed to decode ipc error frame: {}",
            parse_err.format_chain()
        )),
    }
}
