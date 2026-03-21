use std::collections::HashMap;
use std::sync::Arc;

use rwkv_config::raw::infer::GenerationConfig;

use crate::inference_core::InferenceSubmitHandle;
use crate::model_pool::request_router::LoadedModelGroup;

pub fn build_model_groups(
    model_group_engines: HashMap<String, Vec<Arc<InferenceSubmitHandle>>>,
) -> crate::Result<HashMap<String, LoadedModelGroup>> {
    let mut groups = HashMap::new();
    for (model_name, engines) in model_group_engines {
        groups.insert(
            model_name.clone(),
            LoadedModelGroup::new(model_name, engines)?,
        );
    }
    Ok(groups)
}

pub fn build_model_group_engines<F>(
    models: &[GenerationConfig],
    mut build_engines: F,
) -> crate::Result<HashMap<String, LoadedModelGroup>>
where
    F: FnMut(&GenerationConfig) -> crate::Result<Vec<Arc<InferenceSubmitHandle>>>,
{
    let mut model_group_engines: HashMap<String, Vec<Arc<InferenceSubmitHandle>>> = HashMap::new();

    for generation_cfg in models {
        let engines = build_engines(generation_cfg)?;
        model_group_engines
            .entry(generation_cfg.model_name.clone())
            .or_default()
            .extend(engines);
    }

    build_model_groups(model_group_engines)
}
