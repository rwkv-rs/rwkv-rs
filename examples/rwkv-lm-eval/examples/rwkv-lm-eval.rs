use std::path::PathBuf;

use rwkv_config::get_arg_value;
use rwkv_eval::init::init_cfg;
use rwkv_lm_eval::evaluating::evaluating;
use rwkv_lm_eval::paths;

fn resolve_eval_cfg_path(config_dir: &std::path::Path, eval_cfg_name: &str) -> PathBuf {
    [
        config_dir.join("eval").join(format!("{eval_cfg_name}.toml")),
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

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(paths::config_dir);
    let eval_cfg_name = get_arg_value(&args, "--eval-config").unwrap_or_else(|| "example".into());
    let config_path = resolve_eval_cfg_path(&config_dir, &eval_cfg_name);

    let eval_cfg_builder = init_cfg(&config_dir, &eval_cfg_name);

    println!(
        "eval cfg: {eval_cfg_name} (config_dir: {config_dir})",
        config_dir = config_dir.display(),
    );
    println!("config path: {}", config_path.display());
    println!("datasets dir: {}", paths::datasets_path().display());
    println!("logs dir: {}", paths::logs_path().display());

    evaluating(eval_cfg_builder, paths::datasets_path(), config_path, paths::logs_path()).await;
}
