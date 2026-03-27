use std::path::{Path, PathBuf};

use rwkv_config::{load_toml, raw::eval::RawEvalConfig, validated::eval::FinalEvalConfigBuilder};

fn candidate_eval_cfg_paths(config_dir: &Path, eval_cfg_name: &str) -> [PathBuf; 3] {
    [
        config_dir
            .join("eval")
            .join(format!("{eval_cfg_name}.toml")),
        config_dir.join(format!("{eval_cfg_name}.toml")),
        PathBuf::from("examples")
            .join("rwkv-lm-eval")
            .join("config")
            .join(format!("{eval_cfg_name}.toml")),
    ]
}

pub fn init_cfg<P: AsRef<Path>>(config_dir: P, eval_cfg_name: &str) -> FinalEvalConfigBuilder {
    let config_dir = config_dir.as_ref();
    let eval_cfg_path = candidate_eval_cfg_paths(config_dir, eval_cfg_name)
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| {
            panic!(
                "failed to locate eval config `{}` under {}",
                eval_cfg_name,
                config_dir.display()
            )
        });

    let mut raw_eval_cfg: RawEvalConfig = load_toml(&eval_cfg_path);
    raw_eval_cfg.fill_default();

    FinalEvalConfigBuilder::load_from_raw(raw_eval_cfg)
}
