use crate::dtos::health::HealthResp;

pub fn health() -> HealthResp {
    HealthResp {
        status: "ok".to_string(),
    }
}
