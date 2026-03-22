use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthResp {
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::HealthResp;

    #[test]
    fn serializes_health_response() {
        let resp = HealthResp {
            status: "ok".to_string(),
        };
        let json = sonic_rs::to_string(&resp).expect("serialize health response");
        assert_eq!(json, r#"{"status":"ok"}"#);
    }
}
