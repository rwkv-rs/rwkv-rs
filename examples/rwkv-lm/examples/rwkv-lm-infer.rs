#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This example requires feature `inferring`.\nRun: cargo run -p rwkv-lm --example \
         rwkv-lm-infer --no-default-features --features inferring,cuda"
    );
}

#[cfg(feature = "inferring")]
mod infer_main {
    use std::collections::HashMap;
    use std::sync::Arc;

    use rwkv::custom::cubecl::device::DeviceId;
    use rwkv::custom::module::Module;
    use rwkv::custom::prelude::{Backend, DeviceOps};
    use rwkv::custom::record::{FullPrecisionSettings, NamedMpkFileRecorder};
    use rwkv::nn::kernels::rapid_sample::RapidSampleBackend;
    use rwkv::nn::kernels::wkv7_common::Wkv7Backend;

    use rwkv::config::{
        ModelTypeOptions, load_toml, raw::model::RawModelConfig, validated::infer::FinalInferConfig,
    };
    use rwkv::infer::auth::AuthConfig;
    use rwkv::infer::engine::{EngineHandle, EngineRuntime, EngineRuntimeConfig};
    use rwkv::infer::init::{init_cfg, init_log};
    use rwkv::infer::server::{RwkvInferApp, RwkvInferRouterBuilder};
    use rwkv::infer::service::{ModelRuntimeGroup, RwkvInferService};

    use rwkv_lm::inferring::RwkvLmExecutor;
    use rwkv_lm::model::AutoRegressiveModelConfig;

    const INFER_CFG_PATH: &str = "examples/rwkv-lm/config/infer.toml";
    const MODEL_CFG_PATH: &str = "examples/rwkv-lm/config/model.toml";

    #[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
    type ElemType = rwkv::custom::tensor::bf16;
    #[cfg(feature = "f32")]
    type ElemType = f32;
    #[cfg(feature = "flex32")]
    type ElemType = rwkv::custom::tensor::flex32;
    #[cfg(feature = "f16")]
    type ElemType = rwkv::custom::tensor::f16;

    #[derive(Clone, Debug)]
    struct ResolvedModelConfig {
        num_cells: usize,
        vocab_size: usize,
        embedded_dim: usize,
        num_heads: usize,
        head_size: usize,
    }

    fn load_model_cfg() -> ResolvedModelConfig {
        assert!(
            std::path::Path::new(MODEL_CFG_PATH).exists(),
            "model.toml not found: {}",
            MODEL_CFG_PATH
        );
        let mut raw: RawModelConfig = load_toml(MODEL_CFG_PATH);
        raw.fill_default();

        if raw.model_type != ModelTypeOptions::AutoRegressive {
            panic!(
                "Only model_type=auto_regressive is supported for rwkv-lm inference, got {:?}",
                raw.model_type
            );
        }

        assert_eq!(
            raw.embedded_dim % raw.num_heads,
            0,
            "embedded_dim must be divisible by num_heads"
        );

        ResolvedModelConfig {
            num_cells: raw.num_cells,
            vocab_size: raw.vocab_size,
            embedded_dim: raw.embedded_dim,
            num_heads: raw.num_heads,
            head_size: raw.embedded_dim / raw.num_heads,
        }
    }

    fn build_model_group_engines<B>(
        infer_cfg: &FinalInferConfig,
    ) -> rwkv::infer::Result<HashMap<String, ModelRuntimeGroup>>
    where
        B: Backend + Wkv7Backend + RapidSampleBackend,
    {
        let mut model_group_engines: HashMap<String, Vec<Arc<EngineHandle>>> = HashMap::new();

        for model in &infer_cfg.models {
            let max_batch_size = model
                .max_batch_size
                .expect("max_batch_size must be set by infer config default fill");
            let paragraph_len = model
                .paragraph_len
                .expect("paragraph_len must be set by infer config default fill");
            let max_context_len = model
                .max_context_len
                .expect("max_context_len must be set by infer config default fill");
            let decode_first = model
                .decode_first
                .expect("decode_first must be set by infer config default fill");
            let device_type = model
                .device_type
                .expect("device_type must be set by infer config default fill");

            let resolved_model = load_model_cfg();

            for device_id in &model.device_ids {
                let device = B::Device::from_id(DeviceId::new(device_type, *device_id));

                let model_config = AutoRegressiveModelConfig::new(
                    resolved_model.num_cells,
                    resolved_model.vocab_size,
                    resolved_model.embedded_dim,
                    resolved_model.num_heads,
                    resolved_model.head_size,
                );
                let model_runtime = model_config.init::<B>(&device);
                let model_runtime = model_runtime
                    .load_file(
                        &model.weights_path,
                        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                        &device,
                    )
                    .expect("load weights (mpk)");

                let executor = RwkvLmExecutor::<B>::new(
                    device.clone(),
                    model_runtime,
                    max_batch_size,
                    resolved_model.num_cells,
                    resolved_model.vocab_size,
                    resolved_model.embedded_dim,
                    resolved_model.num_heads,
                    resolved_model.head_size,
                );

                let engine = EngineRuntime::spawn(
                    EngineRuntimeConfig {
                        tokenizer_vocab_path: model.tokenizer_vocab_path.clone(),
                        max_batch_size,
                        paragraph_len,
                        max_context_len,
                        decode_first,
                    },
                    Box::new(executor),
                )?;

                log::info!(
                    "engine ready: model_name={} device_type={} device_id={} weights_path={}",
                    model.model_name,
                    device_type,
                    device_id,
                    model.weights_path
                );
                model_group_engines
                    .entry(model.model_name.clone())
                    .or_default()
                    .push(Arc::new(engine));
            }
        }

        let mut groups = HashMap::new();
        for (model_name, engines) in model_group_engines {
            groups.insert(
                model_name.clone(),
                ModelRuntimeGroup::new(model_name, engines)?,
            );
        }
        Ok(groups)
    }

    pub async fn launch<B>() -> rwkv::infer::Result<()>
    where
        B: Backend + Wkv7Backend + RapidSampleBackend,
    {
        let _log_guard = init_log("info");
        let infer_cfg_builder = init_cfg(INFER_CFG_PATH);
        let infer_cfg = infer_cfg_builder.build();

        let groups = build_model_group_engines::<B>(infer_cfg.as_ref())?;
        let service = Arc::new(RwkvInferService::new(groups)?);

        let app = RwkvInferApp {
            cfg: infer_cfg.clone(),
            service,
            auth: AuthConfig {
                api_key: infer_cfg.api_key.clone(),
            },
        };

        let app = RwkvInferRouterBuilder::new().with_app(app).build().await?;

        let listener = tokio::net::TcpListener::bind(&infer_cfg.http_bind_addr)
            .await
            .expect("bind http addr");
        axum::serve(listener, app).await.expect("serve");
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

    pub async fn run() {
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
}

#[cfg(feature = "inferring")]
#[tokio::main]
async fn main() {
    infer_main::run().await;
}
