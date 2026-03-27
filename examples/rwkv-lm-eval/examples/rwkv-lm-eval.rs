use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use rwkv_config::raw::eval::SpaceDbConfig;
use rwkv_config::validated::eval::EVAL_CFG;
use rwkv_eval::init::init_cfg;
use rwkv_lm_eval::config_path::resolve_eval_cfg_path;
use rwkv_lm_eval::db::connect;
use rwkv_lm_eval::http_api::serve;
use rwkv_lm_eval::paths;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_os_t = paths::config_dir())]
    config_dir: PathBuf,
    #[arg(long, default_value = "example")]
    eval_config: String,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long)]
    db_pool_max_connections: Option<u32>,
}

fn validate_space_db_config(cfg: &SpaceDbConfig) -> Result<(), String> {
    if cfg.host.trim().is_empty() {
        return Err("space_db.host cannot be empty".to_string());
    }
    if cfg.username.trim().is_empty() {
        return Err("space_db.username cannot be empty".to_string());
    }
    if cfg.password.trim().is_empty() {
        return Err("space_db.password cannot be empty".to_string());
    }
    if cfg.port.trim().is_empty() {
        return Err("space_db.port cannot be empty".to_string());
    }
    if cfg.database_name.trim().is_empty() {
        return Err("space_db.database_name cannot be empty".to_string());
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let config_path = resolve_eval_cfg_path(&args.config_dir, &args.eval_config);
    let eval_cfg_builder = init_cfg(&args.config_dir, &args.eval_config);
    eval_cfg_builder.build();

    let eval_cfg = EVAL_CFG
        .get()
        .unwrap_or_else(|| panic!("failed to initialize eval config"));
    let space_db_cfg = eval_cfg
        .space_db
        .as_ref()
        .unwrap_or_else(|| panic!("rwkv-lm-eval api requires [space_db] config"));

    validate_space_db_config(space_db_cfg)
        .unwrap_or_else(|err| panic!("invalid [space_db] config: {err}"));

    let max_connections = args
        .db_pool_max_connections
        .unwrap_or(eval_cfg.db_pool_max_connections);

    let db = connect(space_db_cfg, max_connections, false)
        .await
        .unwrap_or_else(|err| panic!("failed to connect to postgres: {err}"));

    println!(
        "rwkv-lm-eval api starting on {} (config: {}, pool_max_connections: {})",
        args.bind,
        config_path.display(),
        max_connections
    );
    println!("config path: {}", config_path.display());
    println!("openapi: http://{}/openapi.json", args.bind);

    serve(args.bind, db)
        .await
        .unwrap_or_else(|err| panic!("http api server failed: {err}"));
}
