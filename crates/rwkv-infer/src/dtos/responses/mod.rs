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

#[cfg(test)]
mod tests {
    use super::{DeleteResp, ResponseIdReq, ResponsesCreateReq, ResponsesResp};

    #[test]
    fn serializes_responses_create_request_with_old_wire_shape() {
        let req = ResponsesCreateReq {
            model: "rwkv".to_string(),
            input: "hello".to_string(),
            background: Some(true),
            stream: Some(false),
            max_output_tokens: Some(32),
            top_k: Some(20),
            temperature: Some(0.7),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            repetition_penalty: Some(1.1),
            penalty_decay: Some(0.996),
            stop: Some(vec!["\n\n".to_string(), "END".to_string()]),
        };

        let json = sonic_rs::to_string(&req).expect("serialize responses request");
        assert!(json.contains(r#""model":"rwkv""#));
        assert!(json.contains(r#""input":"hello""#));
        assert!(json.contains(r#""max_output_tokens":32"#));
        assert!(json.contains(r#""stop":["\n\n","END"]"#));
    }

    #[test]
    fn serializes_responses_resource_shape() {
        let resp = ResponsesResp {
            id: "resp_123".to_string(),
            object: "response".to_string(),
            status: "completed".to_string(),
            output_text: Some("done".to_string()),
        };

        let json = sonic_rs::to_string(&resp).expect("serialize responses response");
        assert_eq!(
            json,
            r#"{"id":"resp_123","object":"response","status":"completed","output_text":"done"}"#
        );
    }

    #[test]
    fn round_trips_response_identifier_and_delete_response() {
        let req: ResponseIdReq =
            sonic_rs::from_str(r#"{"response_id":"resp_123"}"#).expect("deserialize response id");
        assert_eq!(req.response_id, "resp_123");

        let resp = DeleteResp {
            id: "resp_123".to_string(),
            deleted: true,
        };
        let json = sonic_rs::to_string(&resp).expect("serialize delete response");
        assert_eq!(json, r#"{"id":"resp_123","deleted":true}"#);
    }
}
