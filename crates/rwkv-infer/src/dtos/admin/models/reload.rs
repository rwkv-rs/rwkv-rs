use serde::{Deserialize, Serialize};
use rwkv_config::raw::infer::GenerationConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelsReloadReq {
    #[serde(default)]
    pub upsert: Vec<GenerationConfig>,
    #[serde(default)]
    pub remove_model_names: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelsReloadResp {
    pub rebuilt_model_names: Vec<String>,
    pub removed_model_names: Vec<String>,
    pub active_model_names: Vec<String>,
}