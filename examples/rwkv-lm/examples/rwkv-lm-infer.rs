#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This example requires feature `inferring`.\nRun: cargo run -p rwkv-lm --example \
         rwkv-lm-infer --features cuda"
    );
}

use axum::serve;
use rwkv::config::{default_cfg_dir, get_arg_value};
use rwkv::custom::prelude::Backend;
use rwkv::infer::auth::AuthConfig;
#[cfg(feature = "ipc-iceoryx2")]
use rwkv::infer::ipc::IpcServer;
use rwkv::infer::server::{AppState, RouterBuilder};
use rwkv::infer::service::RuntimeManager;
use rwkv::nn::kernels::rapid_sample::RapidSampleBackend;
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

use rwkv_lm::inferring::RwkvLmEngineFactory;
use rwkv_lm::paths;

#[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
type ElemType = rwkv::custom::tensor::bf16;
#[cfg(feature = "f32")]
type ElemType = f32;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;

pub async fn launch<B>() -> rwkv::infer::Result<()>
where
    B: Backend + Wkv7Backend + RapidSampleBackend + Send + Sync + 'static,
{
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_cfg_dir);
    let infer_cfg = get_arg_value(&args, "--infer-cfg").unwrap_or_else(|| "rwkv-lm-7.2b".into());

    let log_dir = paths::logs_dir();
    std::fs::create_dir_all(&log_dir).map_err(|e| {
        rwkv::infer::Error::Internal(format!(
            "failed to create infer log directory {}: {e}",
            log_dir.display()
        ))
    })?;
    #[cfg(not(feature = "trace"))]
    let log_dir_text = log_dir.to_string_lossy().to_string();
    #[cfg(feature = "trace")]
    let trace_mode = rwkv::infer::trace::init_tracing("rwkv-lm-infer")?;

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

    let runtime_manager = Arc::new(RuntimeManager::bootstrap(
        config_dir,
        infer_cfg,
        Arc::new(RwkvLmEngineFactory::<B>::new()),
    )?);

    let app = AppState {
        auth_cfg: AuthConfig {
            api_key: runtime_manager.api_key(),
        },
        runtime_manager: runtime_manager.clone(),
    };

    #[cfg(feature = "ipc-iceoryx2")]
    let _ipc_server_thread = if runtime_manager.ipc_enabled() {
        let server =
            IpcServer::from_runtime_manager(runtime_manager.clone(), app.auth_cfg.clone())?;
        log::info!(
            "starting iceoryx2 ipc service: {}",
            runtime_manager.ipc_service_name()
        );
        Some(server.spawn()?)
    } else {
        None
    };

    #[cfg(not(feature = "ipc-iceoryx2"))]
    if runtime_manager.ipc_enabled() {
        return Err(rwkv::infer::Error::bad_request(
            "ipc is enabled in infer config but feature `ipc-iceoryx2` is not enabled",
        ));
    }

    let router = RouterBuilder::new(app).build().await?;

    let bind_addr = runtime_manager.http_bind_addr();
    let listener = TcpListener::bind(&bind_addr).await.map_err(|e| {
        rwkv::infer::Error::Internal(format!("failed to bind http listener {}: {e}", bind_addr))
    })?;
    serve(listener, router).await.map_err(|e| {
        rwkv::infer::Error::Internal(format!("inference server exited with IO error: {e}"))
    })?;
    Ok(())
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::{ElemType, launch};
    use rwkv::custom::backend::Wgpu;

    pub async fn run() {
        launch::<Wgpu<ElemType, i32>>().await.unwrap();
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::{ElemType, launch};
    use rwkv::custom::backend::Vulkan;

    pub async fn run() {
        launch::<Vulkan<ElemType, i32>>().await.unwrap();
    }
}

#[cfg(feature = "metal")]
mod metal {
    use super::{ElemType, launch};
    use rwkv::custom::backend::Metal;

    pub async fn run() {
        launch::<Metal<ElemType, i32>>().await.unwrap();
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::{ElemType, launch};
    use rwkv::custom::backend::Cuda;

    pub async fn run() {
        launch::<Cuda<ElemType, i32>>().await.unwrap();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use super::{ElemType, launch};
    use rwkv::custom::backend::Rocm;

    pub async fn run() {
        launch::<Rocm<ElemType, i32>>().await.unwrap();
    }
}

#[cfg(feature = "inferring")]
#[tokio::main]
pub async fn main() {
    #[cfg(feature = "wgpu")]
    wgpu::run().await;
    #[cfg(feature = "cuda")]
    cuda::run().await;
    #[cfg(feature = "rocm")]
    rocm::run().await;
    #[cfg(feature = "vulkan")]
    vulkan::run().await;
    #[cfg(feature = "metal")]
    metal::run().await;
}
