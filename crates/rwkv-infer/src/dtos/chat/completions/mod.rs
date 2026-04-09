use serde::{Deserialize, Deserializer, Serialize, de};
use sonic_rs::Value;

use crate::dtos::stop::StopField;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionsReq {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(alias = "frequency_penalty")]
    pub repetition_penalty: Option<f32>,
    pub penalty_decay: Option<f32>,
    pub stop: Option<StopField>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
    pub candidate_token_texts: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub function: FunctionInContext,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionInContext {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchema },
}

impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ResponseFormatHelper {
            #[serde(rename = "type")]
            ty: String,
            #[serde(default)]
            json_schema: Option<JsonSchema>,
        }

        let helper = ResponseFormatHelper::deserialize(deserializer)?;
        match helper.ty.as_str() {
            "text" => Ok(Self::Text),
            "json_object" => Ok(Self::JsonObject),
            "json_schema" => Ok(Self::JsonSchema {
                json_schema: helper
                    .json_schema
                    .ok_or_else(|| de::Error::missing_field("json_schema"))?,
            }),
            other => Err(de::Error::unknown_variant(
                other,
                &["text", "json_object", "json_schema"],
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: Function,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(NamedToolChoice),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: NamedToolChoiceFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedToolChoiceFunction {
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionResp {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choices>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choices {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[cfg(test)]
mod tests {
    use super::{ChatCompletionsReq, ResponseFormat};

    #[test]
    fn parses_response_format_json_schema_with_schema_value() {
        let req: ChatCompletionsReq = sonic_rs::from_str(
            r#"{
                "model":"test-model",
                "messages":[{"role":"user","content":"hi"}],
                "response_format":{
                    "type":"json_schema",
                    "json_schema":{
                        "name":"result",
                        "schema":{"type":"object","properties":{},"additionalProperties":false}
                    }
                }
            }"#,
        )
        .expect("parse chat completions request");

        match req.response_format.expect("response format") {
            ResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "result");
                assert_eq!(
                    json_schema.schema.expect("schema"),
                    sonic_rs::json!({"type":"object","properties":{},"additionalProperties":false})
                );
            }
            other => panic!("unexpected response format: {other:?}"),
        }
    }

    #[test]
    fn parses_frequency_penalty_into_repetition_penalty() {
        let req: ChatCompletionsReq = sonic_rs::from_str(
            r#"{
                "model":"test-model",
                "messages":[{"role":"user","content":"hi"}],
                "frequency_penalty":0.25
            }"#,
        )
        .expect("parse chat completions request");

        assert_eq!(req.repetition_penalty, Some(0.25));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Logprobs {
    pub content: Vec<Content>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Content {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprobs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopLogprobs {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionsChunkResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub ty: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ChunkToolCallFunction>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChunkToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
