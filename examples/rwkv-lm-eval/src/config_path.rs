use std::path::{Path, PathBuf};

pub fn resolve_eval_cfg_path(config_dir: &Path, eval_cfg_name: &str) -> PathBuf {
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
    .into_iter()
    .find(|path| path.is_file())
    .unwrap_or_else(|| {
        panic!(
            "failed to locate eval config `{}` under {}",
            eval_cfg_name,
            config_dir.display()
        )
    })
}
