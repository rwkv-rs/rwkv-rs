use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use rwkv_config::load_toml;
use rwkv_config::raw::eval::{RawEvalConfig, SpaceDbConfig};
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
    let mut raw_eval_cfg: RawEvalConfig = load_toml(&config_path);
    raw_eval_cfg.fill_default();

    validate_space_db_config(&raw_eval_cfg.space_db)
        .unwrap_or_else(|err| panic!("invalid [space_db] config: {err}"));

    let max_connections = args
        .db_pool_max_connections
        .or(raw_eval_cfg.db_pool_max_connections)
        .unwrap_or(32);

    let db = connect(&raw_eval_cfg.space_db, max_connections)
        .await
        .unwrap_or_else(|err| panic!("failed to connect to postgres: {err}"));

    println!(
        "rwkv-lm-eval api starting on {} (config: {}, pool_max_connections: {})",
        args.bind,
        config_path.display(),
        max_connections
    );
    println!("openapi: http://{}/openapi.json", args.bind);

    serve(args.bind, db)
        .await
        .unwrap_or_else(|err| panic!("http api server failed: {err}"));
}
