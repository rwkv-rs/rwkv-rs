use rwkv_config::raw::eval::IntApiConfig;
use rwkv_config::validated::eval::EVAL_CFG;

pub(crate) fn collect_models() -> Vec<IntApiConfig> {
    let mut target_models = Vec::new();
    for model_arch_version in &EVAL_CFG.get().unwrap().model_arch_versions {
        for model_data_version in &EVAL_CFG.get().unwrap().model_data_versions {
            for model_num_param in &EVAL_CFG.get().unwrap().model_num_params {
                target_models.extend(
                    EVAL_CFG
                        .get()
                        .unwrap()
                        .models
                        .iter()
                        .filter(|model| {
                            model.model_arch_version == *model_arch_version
                                && model.model_data_version == *model_data_version
                                && model.model_num_params == *model_num_param
                        })
                        .cloned(),
                );
            }
        }
    }

    target_models
}

pub(crate) fn model_cache_key(api_cfg: &IntApiConfig) -> String {
    format!(
        "{}|{}|{}|{}",
        api_cfg.model_arch_version,
        api_cfg.model_data_version,
        api_cfg.model_num_params,
        api_cfg.model
    )
}
