use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use uuid::Uuid;

use crate::{dtos::errors::OpenAiErrorResponse, services::ServiceError};

pub const IPC_PROTOCOL_VERSION: u16 = 1;

const REQUEST_MAGIC: u32 = 0x4957_4B52;
const RESPONSE_MAGIC: u32 = 0x5257_4B52;
const REQUEST_HEADER_LEN: usize = 36;
const RESPONSE_HEADER_LEN: usize = 32;
const FLAG_STREAM: u16 = 0x1;
const FLAG_HAS_API_KEY: u16 = 0x2;

pub type IpcResult<T> = Result<T, IpcError>;

#[derive(Clone, Debug, Error)]
pub enum IpcError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Unauthorized(String),
    #[error("{0}")]
    NotSupported(String),
    #[error("{0}")]
    Internal(String),
}

impl IpcError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::Unauthorized(message.into())
    }

    pub fn not_supported(message: impl Into<String>) -> Self {
        Self::NotSupported(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::Unauthorized(_) => 401,
            Self::NotSupported(_) => 501,
            Self::Internal(_) => 500,
        }
    }

    pub fn openai_error_response(&self) -> OpenAiErrorResponse {
        match self {
            Self::BadRequest(message) => OpenAiErrorResponse::bad_request(message.clone()),
            Self::Unauthorized(message) => OpenAiErrorResponse::unauthorized(message.clone()),
            Self::NotSupported(message) => OpenAiErrorResponse::not_supported(message.clone()),
            Self::Internal(message) => OpenAiErrorResponse::internal(message.clone()),
        }
    }

    pub fn from_status_and_body(status_code: u16, body: OpenAiErrorResponse) -> Self {
        let message = body.error.message;
        if status_code == 401 {
            Self::Unauthorized(message)
        } else if body.error.ty == "not_supported" || status_code == 501 {
            Self::NotSupported(message)
        } else if status_code >= 500 {
            Self::Internal(message)
        } else {
            Self::BadRequest(message)
        }
    }
}

impl From<ServiceError> for IpcError {
    fn from(value: ServiceError) -> Self {
        let (status, body) = value.into_parts();
        Self::from_status_and_body(status.as_u16(), body)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum RouteId {
    Handshake = 0,
    Health = 1,
    ModelsList = 2,
    Completions = 3,
    ChatCompletions = 4,
    ResponsesCreate = 5,
    ResponsesGet = 6,
    ResponsesDelete = 7,
    ResponsesCancel = 8,
    Embeddings = 9,
    ImagesGenerations = 10,
    AudioSpeech = 11,
    AdminModelsReload = 12,
}

impl RouteId {
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Handshake),
            1 => Some(Self::Health),
            2 => Some(Self::ModelsList),
            3 => Some(Self::Completions),
            4 => Some(Self::ChatCompletions),
            5 => Some(Self::ResponsesCreate),
            6 => Some(Self::ResponsesGet),
            7 => Some(Self::ResponsesDelete),
            8 => Some(Self::ResponsesCancel),
            9 => Some(Self::Embeddings),
            10 => Some(Self::ImagesGenerations),
            11 => Some(Self::AudioSpeech),
            12 => Some(Self::AdminModelsReload),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResponseKind {
    Data = 1,
    Done = 2,
    Error = 3,
}

impl ResponseKind {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Data),
            2 => Some(Self::Done),
            3 => Some(Self::Error),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IpcRequest {
    pub version: u16,
    pub request_id: Uuid,
    pub route: RouteId,
    pub stream: bool,
    pub api_key: Option<String>,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct IpcResponse {
    pub version: u16,
    pub request_id: Uuid,
    pub route: RouteId,
    pub kind: ResponseKind,
    pub status_code: u16,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: u16,
    pub client_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub accepted: bool,
    pub server_version: u16,
    pub reason: Option<String>,
}

pub fn encode_request(req: &IpcRequest) -> IpcResult<Vec<u8>> {
    let api_key = req.api_key.as_deref().unwrap_or("").as_bytes();
    let api_key_len = api_key.len();
    let payload_len = req.payload.len();
    if api_key_len > u32::MAX as usize {
        return Err(IpcError::bad_request(
            "ipc request api key exceeds u32 max length",
        ));
    }
    if payload_len > u32::MAX as usize {
        return Err(IpcError::bad_request(
            "ipc request payload exceeds u32 max length",
        ));
    }

    let mut out = Vec::with_capacity(REQUEST_HEADER_LEN + api_key_len + payload_len);
    out.extend_from_slice(&REQUEST_MAGIC.to_le_bytes());
    out.extend_from_slice(&req.version.to_le_bytes());
    out.extend_from_slice(&(req.route as u16).to_le_bytes());
    let mut flags = 0u16;
    if req.stream {
        flags |= FLAG_STREAM;
    }
    if req.api_key.is_some() {
        flags |= FLAG_HAS_API_KEY;
    }
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(req.request_id.as_bytes());
    out.extend_from_slice(&(api_key_len as u32).to_le_bytes());
    out.extend_from_slice(&(payload_len as u32).to_le_bytes());
    out.extend_from_slice(api_key);
    out.extend_from_slice(&req.payload);
    Ok(out)
}

pub fn decode_request(input: &[u8]) -> IpcResult<IpcRequest> {
    if input.len() < REQUEST_HEADER_LEN {
        return Err(IpcError::bad_request("ipc request too short"));
    }

    let magic = u32::from_le_bytes(read_array::<4>(input, 0)?);
    if magic != REQUEST_MAGIC {
        return Err(IpcError::bad_request("ipc request magic mismatch"));
    }

    let version = u16::from_le_bytes(read_array::<2>(input, 4)?);
    let route_raw = u16::from_le_bytes(read_array::<2>(input, 6)?);
    let route = RouteId::from_u16(route_raw)
        .ok_or_else(|| IpcError::bad_request(format!("unknown ipc route id {route_raw}")))?;
    let flags = u16::from_le_bytes(read_array::<2>(input, 8)?);
    let request_id = Uuid::from_slice(&read_array::<16>(input, 12)?)
        .map_err(|e| IpcError::bad_request(format!("invalid request id: {e}")))?;
    let api_key_len = u32::from_le_bytes(read_array::<4>(input, 28)?) as usize;
    let payload_len = u32::from_le_bytes(read_array::<4>(input, 32)?) as usize;

    let expected = REQUEST_HEADER_LEN
        .checked_add(api_key_len)
        .and_then(|n| n.checked_add(payload_len))
        .ok_or_else(|| IpcError::bad_request("ipc request length overflow"))?;
    if input.len() != expected {
        return Err(IpcError::bad_request(format!(
            "ipc request length mismatch: expected {expected}, got {}",
            input.len()
        )));
    }

    let mut cursor = REQUEST_HEADER_LEN;
    let api_key = if flags & FLAG_HAS_API_KEY != 0 {
        let raw = &input[cursor..cursor + api_key_len];
        cursor += api_key_len;
        Some(
            std::str::from_utf8(raw)
                .map_err(|e| IpcError::bad_request(format!("invalid api key utf8: {e}")))?
                .to_string(),
        )
    } else {
        None
    };
    let payload = input[cursor..cursor + payload_len].to_vec();

    Ok(IpcRequest {
        version,
        request_id,
        route,
        stream: flags & FLAG_STREAM != 0,
        api_key,
        payload,
    })
}

pub fn encode_response(resp: &IpcResponse) -> IpcResult<Vec<u8>> {
    let payload_len = resp.payload.len();
    if payload_len > u32::MAX as usize {
        return Err(IpcError::bad_request(
            "ipc response payload exceeds u32 max length",
        ));
    }

    let mut out = Vec::with_capacity(RESPONSE_HEADER_LEN + payload_len);
    out.extend_from_slice(&RESPONSE_MAGIC.to_le_bytes());
    out.extend_from_slice(&resp.version.to_le_bytes());
    out.extend_from_slice(&(resp.route as u16).to_le_bytes());
    out.push(resp.kind as u8);
    out.push(0);
    out.extend_from_slice(&resp.status_code.to_le_bytes());
    out.extend_from_slice(resp.request_id.as_bytes());
    out.extend_from_slice(&(payload_len as u32).to_le_bytes());
    out.extend_from_slice(&resp.payload);
    Ok(out)
}

pub fn decode_response(input: &[u8]) -> IpcResult<IpcResponse> {
    if input.len() < RESPONSE_HEADER_LEN {
        return Err(IpcError::bad_request("ipc response too short"));
    }

    let magic = u32::from_le_bytes(read_array::<4>(input, 0)?);
    if magic != RESPONSE_MAGIC {
        return Err(IpcError::bad_request("ipc response magic mismatch"));
    }
    let version = u16::from_le_bytes(read_array::<2>(input, 4)?);
    let route_raw = u16::from_le_bytes(read_array::<2>(input, 6)?);
    let route = RouteId::from_u16(route_raw)
        .ok_or_else(|| IpcError::bad_request(format!("unknown ipc route id {route_raw}")))?;
    let kind_raw = input[8];
    let kind = ResponseKind::from_u8(kind_raw)
        .ok_or_else(|| IpcError::bad_request(format!("unknown ipc response kind {kind_raw}")))?;
    let status_code = u16::from_le_bytes(read_array::<2>(input, 10)?);
    let request_id = Uuid::from_slice(&read_array::<16>(input, 12)?)
        .map_err(|e| IpcError::bad_request(format!("invalid request id: {e}")))?;
    let payload_len = u32::from_le_bytes(read_array::<4>(input, 28)?) as usize;

    let expected = RESPONSE_HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| IpcError::bad_request("ipc response length overflow"))?;
    if input.len() != expected {
        return Err(IpcError::bad_request(format!(
            "ipc response length mismatch: expected {expected}, got {}",
            input.len()
        )));
    }

    Ok(IpcResponse {
        version,
        request_id,
        route,
        kind,
        status_code,
        payload: input[RESPONSE_HEADER_LEN..].to_vec(),
    })
}

pub fn encode_json<T: Serialize>(value: &T) -> IpcResult<Vec<u8>> {
    sonic_rs::to_vec(value)
        .map_err(|e| IpcError::bad_request(format!("failed to serialize ipc json: {e}")))
}

pub fn decode_json<T: DeserializeOwned>(payload: &[u8], context: &str) -> IpcResult<T> {
    sonic_rs::from_slice(payload)
        .map_err(|e| IpcError::bad_request(format!("invalid {context}: {e}")))
}

fn read_array<const N: usize>(input: &[u8], offset: usize) -> IpcResult<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| IpcError::bad_request("offset overflow"))?;
    if end > input.len() {
        return Err(IpcError::bad_request("buffer too short"));
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&input[offset..end]);
    Ok(out)
}

