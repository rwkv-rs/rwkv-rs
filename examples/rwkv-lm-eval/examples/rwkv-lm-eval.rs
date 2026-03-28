use std::{net::SocketAddr, path::PathBuf};

use axum::serve;
use clap::Parser;
use rwkv_eval::routes::build_router;
use rwkv_lm_eval::{init::build_http_runtime, paths};
use tokio::net::TcpListener;

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

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let (app_state, config_path, max_connections) = build_http_runtime(
        &args.config_dir,
        &args.eval_config,
        args.db_pool_max_connections,
    )
    .await;

    println!(
        "rwkv-lm-eval api starting on {} (config: {}, pool_max_connections: {})",
        args.bind,
        config_path.display(),
        max_connections
    );
    println!("openapi: http://{}/openapi.json", args.bind);

    let router = build_router(app_state);
    let listener = TcpListener::bind(&args.bind)
        .await
        .unwrap_or_else(|err| panic!("failed to bind http listener {}: {err}", args.bind));
    serve(listener, router)
        .await
        .unwrap_or_else(|err| panic!("http api server failed: {err}"));
}
