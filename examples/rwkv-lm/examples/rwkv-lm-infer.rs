#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This example requires feature `inferring`.\nRun: cargo run -p rwkv-lm --example \
         rwkv-lm-infer --features cuda"
    );
}

use std::{fs::create_dir_all, path::PathBuf};

use axum::serve;
use rwkv::{
    config::{default_cfg_dir, get_arg_value},
    custom::prelude::Backend,
    infer::routes::HttpApiRouterBuilder,
    nn::kernels::{
        addcmul::AddcmulBackend,
        rapid_sample::RapidSampleBackend,
        token_shift_diff::TokenShiftDiffBackend,
        wkv7_common::Wkv7Backend,
    },
};
#[cfg(feature = "ipc")]
use rwkv::infer::routes::IpcServer;
use rwkv_lm::{inferring::build_http_runtime, paths};
#[cfg(feature = "ipc")]
use rwkv_lm::inferring::build_ipc_server_config;
#[cfg(feature = "trace")]
use rwkv_bench::trace::init_tracing;
use tokio::net::TcpListener;

#[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
type ElemType = rwkv::custom::tensor::bf16;
#[cfg(feature = "f32")]
type ElemType = f32;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;

pub async fn launch<B: Backend>()
where
    B: Wkv7Backend
        + TokenShiftDiffBackend
        + AddcmulBackend
        + RapidSampleBackend
        + Send
        + Sync
        + 'static,
{
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_cfg_dir);
    let infer_cfg =
        get_arg_value(&args, "--infer-cfg").unwrap_or_else(|| "rwkv-7.2b-g1e".to_string());

    let log_dir = paths::logs_dir();
    create_dir_all(&log_dir).unwrap_or_else(|e| {
        panic!(
            "failed to create infer log directory {}: {e}",
            log_dir.display()
        )
    });
    #[cfg(not(feature = "trace"))]
    let log_dir_text = log_dir.to_string_lossy().to_string();
    #[cfg(feature = "trace")]
    let trace_mode = init_tracing("rwkv-lm-infer").unwrap();

    #[cfg(feature = "trace")]
    let _log_guard: Option<clia_tracing_config::WorkerGuard> = None;
    #[cfg(not(feature = "trace"))]
    let _log_guard = Some(
        clia_tracing_config::build()
            .filter_level("info")
            .with_ansi(true)
            .to_stdout(true)
            .directory(&log_dir_text)
            .file_name("infer.log")
            .init(),
    );
    println!(
        "infer cfg: {infer_cfg} (config_dir: {})",
        config_dir.display()
    );
    println!("infer logs: {}", log_dir.display());
    #[cfg(feature = "trace")]
    println!("trace mode: {trace_mode:?}");

    let (app_state, bind_addr) = build_http_runtime::<B>(config_dir.clone(), &infer_cfg);

    #[cfg(feature = "ipc")]
    if let Some(ipc_config) = build_ipc_server_config(config_dir.clone(), &infer_cfg) {
        let ipc_server = IpcServer::new(
            ipc_config.clone(),
            app_state.auth_cfg.clone(),
            app_state.queues.clone(),
            app_state.gpu_metrics.clone(),
            app_state.reload_lock.clone(),
            app_state.infer_cfg_path.clone(),
            app_state.build_queues.clone(),
        )
        .unwrap_or_else(|e| {
            panic!(
                "failed to create ipc server {}: {e}",
                ipc_config.service_name
            )
        });
        ipc_server.spawn().unwrap_or_else(|e| {
            panic!(
                "failed to spawn ipc server {}: {e}",
                ipc_config.service_name
            )
        });
        println!("ipc service: {}", ipc_config.service_name);
    }

    let router = HttpApiRouterBuilder::new(app_state).build().await;

    let listener = TcpListener::bind(&bind_addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind http listener {}: {e}", bind_addr));
    serve(listener, router)
        .await
        .unwrap_or_else(|e| panic!("inference server exited with IO error: {e}"));
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use rwkv::custom::backend::Wgpu;

    use super::{ElemType, launch};

    pub async fn run() {
        launch::<Wgpu<ElemType, i32>>().await;
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use rwkv::custom::backend::Vulkan;

    use super::{ElemType, launch};

    pub async fn run() {
        launch::<Vulkan<ElemType, i32>>().await;
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use rwkv::custom::backend::Cuda;

    use super::{ElemType, launch};

    pub async fn run() {
        launch::<Cuda<ElemType, i32>>().await;
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use rwkv::custom::backend::Hip;

    use super::{ElemType, launch};

    pub async fn run() {
        launch::<Hip<ElemType, i32>>().await;
    }
}

#[cfg(feature = "metal")]
mod metal {
    use rwkv::custom::backend::Metal;

    use super::{ElemType, launch};

    pub async fn run() {
        launch::<Metal<ElemType, i32>>().await;
    }
}

#[tokio::main]
async fn main() {
    #[cfg(feature = "wgpu")]
    wgpu::run().await;
    #[cfg(feature = "vulkan")]
    vulkan::run().await;
    #[cfg(feature = "cuda")]
    cuda::run().await;
    #[cfg(feature = "rocm")]
    rocm::run().await;
    #[cfg(feature = "metal")]
    metal::run().await;
}
