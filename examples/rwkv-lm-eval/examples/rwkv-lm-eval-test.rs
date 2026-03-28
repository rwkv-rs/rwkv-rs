use std::path::PathBuf;

use rwkv_config::get_arg_value;
use rwkv_eval::sdk::RwkvEvalClient;
use rwkv_lm_eval::{
    init::{build_eval_cfg_builder, resolve_eval_cfg_path},
    paths,
};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(paths::config_dir);
    let eval_cfg_name = get_arg_value(&args, "--eval-config").unwrap_or_else(|| "example".into());
    let client = RwkvEvalClient::new();

    println!(
        "eval cfg: {eval_cfg_name} (config_dir: {config_dir})",
        config_dir = config_dir.display(),
    );
    println!("datasets dir: {}", paths::datasets_path().display());
    println!("logs dir: {}", paths::logs_path().display());

    let config_path = resolve_eval_cfg_path(&config_dir, &eval_cfg_name);
    let eval_cfg_builder = build_eval_cfg_builder(&config_path);

    client
        .run(
            eval_cfg_builder,
            paths::datasets_path(),
            config_path,
            paths::logs_path(),
        )
        .await
        .unwrap_or_else(|err| panic!("local eval sdk failed: {}", err.message()));
}
