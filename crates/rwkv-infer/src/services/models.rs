use crate::{
    dtos::models::{ModelObject, ModelsResp},
    services::QueueMap,
};

pub fn models(queues: &QueueMap) -> ModelsResp {
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
