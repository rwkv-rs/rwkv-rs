#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This example requires feature `inferring`.\nRun: cargo run -p rwkv-lm --example \
         rwkv-lm-infer --no-default-features --features inferring,cuda"
    );
}

use axum::serve;
use rwkv::config::raw::infer::GenerationConfig;
use rwkv::config::validated::model::FinalModelConfig;
use rwkv::config::{default_cfg_dir, get_arg_value};
use rwkv::custom::cubecl::device::DeviceId;
use rwkv::custom::module::Module;
use rwkv::custom::prelude::{Backend, DeviceOps};
use rwkv::custom::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rwkv::infer::auth::AuthConfig;
use rwkv::infer::engine::{EngineRuntime, EngineRuntimeConfig};
use rwkv::infer::init::init_log;
#[cfg(feature = "ipc-iceoryx2")]
use rwkv::infer::ipc::IpcServer;
use rwkv::infer::server::{AppState, RouterBuilder};
use rwkv::infer::service::builder::build_model_group_engines;
use rwkv::infer::service::{ModelEngineFactory, ModelRuntimeGroup, RuntimeManager};
use rwkv::nn::kernels::rapid_sample::RapidSampleBackend;
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

use rwkv_lm::inferring::RwkvLmExecutor;
use rwkv_lm::model::AutoRegressiveModelConfig;

#[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
type ElemType = rwkv::custom::tensor::bf16;
#[cfg(feature = "f32")]
type ElemType = f32;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;

fn required_field<T: Copy>(
    value: Option<T>,
    field: &str,
    model_name: &str,
) -> rwkv::infer::Result<T>
where
    T: Copy,
{
    value.ok_or_else(|| {
        rwkv::infer::Error::BadRequest(format!(
            "missing {} for model {} in infer config",
            field, model_name
        ))
    })
}

struct RwkvLmEngineFactory<B> {
    _marker: PhantomData<B>,
}

impl<B> RwkvLmEngineFactory<B> {
    fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B> ModelEngineFactory for RwkvLmEngineFactory<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend + Send + Sync,
{
    fn build_model_groups(
        &self,
        models: &[GenerationConfig],
        model_cfgs: &HashMap<String, Arc<FinalModelConfig>>,
    ) -> rwkv::infer::Result<HashMap<String, ModelRuntimeGroup>> {
        build_model_group_engines(models, |generation_cfg| {
            let model_cfg = model_cfgs.get(&generation_cfg.model_name).ok_or_else(|| {
                rwkv::infer::Error::Internal(format!(
                    "missing model cfg for {}",
                    generation_cfg.model_name
                ))
            })?;

            let device_type = required_field(
                generation_cfg.device_type,
                "device_type",
                &generation_cfg.model_name,
            )?;
            let max_batch_size = required_field(
                generation_cfg.max_batch_size,
                "max_batch_size",
                &generation_cfg.model_name,
            )?;
            let paragraph_len = required_field(
                generation_cfg.paragraph_len,
                "paragraph_len",
                &generation_cfg.model_name,
            )?;
            let max_context_len = required_field(
                generation_cfg.max_context_len,
                "max_context_len",
                &generation_cfg.model_name,
            )?;
            let decode_first = required_field(
                generation_cfg.decode_first,
                "decode_first",
                &generation_cfg.model_name,
            )?;

            let mut engines = Vec::new();
            for device_id in &generation_cfg.device_ids {
                let device = B::Device::from_id(DeviceId::new(device_type, *device_id));

                let model_config = AutoRegressiveModelConfig::new(
                    model_cfg.num_cells,
                    model_cfg.vocab_size,
                    model_cfg.embedded_dim,
                    model_cfg.num_heads,
                    model_cfg.head_size_auto,
                );
                let model_runtime = model_config.init::<B>(&device);
                let model_runtime = model_runtime
                    .load_file(
                        &generation_cfg.weights_path,
                        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                        &device,
                    )
                    .map_err(|e| {
                        rwkv::infer::Error::Internal(format!(
                            "failed to load weights {} for model {}: {e}",
                            generation_cfg.weights_path, generation_cfg.model_name
                        ))
                    })?;

                let executor = RwkvLmExecutor::<B>::new(
                    device.clone(),
                    model_runtime,
                    model_cfg.clone(),
                    max_batch_size,
                );

                let engine = EngineRuntime::spawn(
                    EngineRuntimeConfig {
                        tokenizer_vocab_path: generation_cfg.tokenizer_vocab_path.clone(),
                        max_batch_size,
                        paragraph_len,
                        max_context_len,
                        decode_first,
                    },
                    Box::new(executor),
                )?;

                log::info!(
                    "engine ready: model_name={} model_cfg={} device_type={} device_id={} \
                     weights_path={}",
                    generation_cfg.model_name,
                    generation_cfg.model_cfg,
                    device_type,
                    device_id,
                    generation_cfg.weights_path
                );
                engines.push(Arc::new(engine));
            }

            Ok(engines)
        })
    }
}

pub async fn launch<B>() -> rwkv::infer::Result<()>
where
    B: Backend + Wkv7Backend + RapidSampleBackend + Send + Sync + 'static,
{
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_cfg_dir);
    let infer_cfg = get_arg_value(&args, "--infer-cfg").unwrap_or_else(|| "rwkv-lm-7.2b".into());

    let _log_guard = init_log("info");
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
