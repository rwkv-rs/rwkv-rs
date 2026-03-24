use crate::dtos::models::{ModelObject, ModelsResp};
use crate::routes::AppState;

pub async fn models(state: AppState) -> ModelsResp {
    let queues = state.queues.read().await;
    let mut data: Vec<ModelObject> = queues
        .keys()
        .cloned()
        .map(|id| ModelObject {
            id,
            object: "model".to_string(),
            owned_by: "rwkv-rs".to_string(),
        })
        .collect();
    data.sort_by(|left, right| left.id.cmp(&right.id));

    ModelsResp {
        object: "list".to_string(),
        data,
    }
}
