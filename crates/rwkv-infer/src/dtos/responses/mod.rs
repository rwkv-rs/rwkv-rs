use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesCreateReq {
    pub model: String,
    pub input: String,
    pub background: Option<bool>,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub top_k: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub penalty_decay: Option<f32>,
    pub stop: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponsesResp {
    pub id: String,
    pub object: String,
    pub status: String,
    pub output_text: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseIdReq {
    pub response_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeleteResp {
    pub id: String,
    pub deleted: bool,
}

