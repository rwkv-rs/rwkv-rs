use std::collections::HashMap;
use std::sync::Arc;

use rwkv_config::raw::infer::GenerationConfig;

use crate::engine::EngineHandle;

use super::ModelRuntimeGroup;

pub fn build_model_groups(
    model_group_engines: HashMap<String, Vec<Arc<EngineHandle>>>,
) -> crate::Result<HashMap<String, ModelRuntimeGroup>> {
    let mut groups = HashMap::new();
    for (model_name, engines) in model_group_engines {
        groups.insert(
            model_name.clone(),
            ModelRuntimeGroup::new(model_name, engines)?,
        );
    }
    Ok(groups)
}

pub fn build_model_group_engines<F>(
    models: &[GenerationConfig],
    mut build_engines: F,
) -> crate::Result<HashMap<String, ModelRuntimeGroup>>
where
    F: FnMut(&GenerationConfig) -> crate::Result<Vec<Arc<EngineHandle>>>,
{
    let mut model_group_engines: HashMap<String, Vec<Arc<EngineHandle>>> = HashMap::new();

    for generation_cfg in models {
        let engines = build_engines(generation_cfg)?;
        model_group_engines
            .entry(generation_cfg.model_name.clone())
            .or_default()
            .extend(engines);
    }

    build_model_groups(model_group_engines)
}
