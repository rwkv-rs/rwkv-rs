#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This example requires feature `inferring`.\nRun: cargo run -p rwkv-lm --example \
         rwkv-lm-infer --features cuda"
    );
}

use std::{fs::create_dir_all, net::SocketAddr, path::PathBuf, sync::Arc};

use axum::serve;
use tokio::net::TcpListener;
use rwkv::{
    config::{default_cfg_dir, get_arg_value},
    custom::prelude::Backend,
    infer::{
        access::http_api::{HttpApiRouterBuilder, HttpApiState},
        auth::AuthConfig,
        model_pool::loaded_model_registry::LoadedModelRegistry,
    },
    nn::kernels::{
        addcmul::AddcmulBackend, rapid_sample::RapidSampleBackend,
        token_shift_diff::TokenShiftDiffBackend, wkv7_common::Wkv7Backend,
    },
};
use rwkv_lm::{inferring::RwkvLmEngineFactory, paths};
#[cfg(feature = "trace")]
use log::info;
#[cfg(feature = "ipc")]
use rwkv::infer::access::ipc_api::IpcServer;
#[cfg(feature = "trace")]
use rwkv_bench::trace::init_tracing;

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

    let loaded_model_registry = Arc::new(LoadedModelRegistry::bootstrap(
        config_dir,
        infer_cfg,
        Arc::new(RwkvLmEngineFactory::<B>::new()),
    ));

    let app = HttpApiState {
        auth_cfg: AuthConfig {
            api_key: loaded_model_registry.api_key(),
        },
        runtime_manager: loaded_model_registry.clone(),
    };

    #[cfg(feature = "ipc")]
    let _ipc_server_thread = if loaded_model_registry.ipc_enabled() {
        let server =
            IpcServer::from_runtime_manager(loaded_model_registry.clone(), app.auth_cfg.clone())
                .unwrap();
        info!(
            "starting ipc service: {}",
            loaded_model_registry.ipc_service_name()
        );
        Some(server.spawn().unwrap())
    } else {
        None
    };

    #[cfg(not(feature = "ipc"))]
    assert!(
        !loaded_model_registry.ipc_enabled(),
        "ipc is enabled in infer config but feature `ipc` is not enabled"
    );

    let router = HttpApiRouterBuilder::new(app).build().await.unwrap();

    let bind_addr: SocketAddr = loaded_model_registry
        .http_bind_addr()
        .parse()
        .unwrap_or_else(|e| panic!("invalid http bind addr: {e}"));
    let listener = TcpListener::bind(bind_addr)
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
