#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This example requires feature `infer`.\nRun: cargo run -p rwkv-lm --example \
         rwkv-lm-infer --no-default-features --features infer,cuda"
    );
}

#[cfg(feature = "inferring")]
mod infer_main {
    use std::sync::Arc;

    use rwkv::custom::cubecl::device::DeviceId;
    use rwkv::custom::module::Module;
    use rwkv::custom::prelude::{Backend, DeviceOps};
    use rwkv::custom::record::{FullPrecisionSettings, NamedMpkFileRecorder};
    use rwkv::nn::kernels::rapid_sample::RapidSampleBackend;
    use rwkv::nn::kernels::wkv7_common::Wkv7Backend;

    use rwkv::config::{
        ModelTypeOptions, load_toml,
        raw::{infer::RawInferConfig, model::RawModelConfig},
        validated::{
            infer::{FinalInferConfig, FinalInferConfigBuilder},
            model::{FinalModelConfigBuilder, MODEL_CFG},
        },
    };
    use rwkv_infer::auth::AuthConfig;
    use rwkv_infer::config::BackendConfig;
    use rwkv_infer::engine::{EngineRuntime, EngineRuntimeConfig};
    use rwkv_infer::server::{RwkvInferRouterBuilder, SharedRwkvInferState};
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

    fn load_infer_config(path: &str) -> Arc<FinalInferConfig> {
        let mut raw: RawInferConfig = load_toml(path);
        raw.fill_default();
        let builder = FinalInferConfigBuilder::load_from_raw(raw);
        builder.check();
        builder.build()
    }

    fn load_model_config(path: &str) {
        let mut raw: RawModelConfig = load_toml(path);
        raw.fill_default();
        let mut builder = FinalModelConfigBuilder::load_from_raw(raw);
        builder.fill_auto_after_load();
        builder.build();
    }

    pub async fn launch<B>()
    where
        B: Backend + Wkv7Backend + RapidSampleBackend,
    {
        let infer_cfg_path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "examples/rwkv-lm/config/infer.toml".to_string());

        let _guard = clia_tracing_config::build()
            .filter_level("info")
            .with_ansi(true)
            .to_stdout(true)
            .init();

        let infer_cfg = load_infer_config(&infer_cfg_path);
        load_model_config(&infer_cfg.model_config_path);

        let model_cfg = MODEL_CFG.get().expect("MODEL_CFG must be built").clone();
        if model_cfg.model_type != ModelTypeOptions::AutoRegressive {
            panic!("Only model_type=auto_regressive is supported for rwkv-lm inference example");
        }

        let device = B::Device::from_id(DeviceId::new(
            infer_cfg.device_id_type,
            infer_cfg.device_id_index,
        ));

        let model = AutoRegressiveModelConfig::new(
            model_cfg.num_cells,
            model_cfg.vocab_size,
            model_cfg.embedded_dim,
            model_cfg.num_heads,
            model_cfg.head_size_auto,
        )
        .init::<B>(&device);

        let model = model
            .load_file(
                &infer_cfg.weights_path,
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                &device,
            )
            .expect("load weights (mpk)");

        let executor = RwkvLmExecutor::<B>::new(
            device.clone(),
            &infer_cfg.tokenizer_vocab_path,
            model,
            infer_cfg.max_batch_size,
            model_cfg.num_cells,
            model_cfg.vocab_size,
            model_cfg.embedded_dim,
            model_cfg.num_heads,
            model_cfg.head_size_auto,
        );

        let backend_cfg = BackendConfig::from(infer_cfg.as_ref());

        let engine = EngineRuntime::spawn(
            EngineRuntimeConfig {
                backend: backend_cfg.clone(),
            },
            Box::new(executor),
        );

        let state = SharedRwkvInferState {
            cfg: backend_cfg.clone(),
            engine: Arc::new(engine),
            auth: AuthConfig {
                api_key: infer_cfg.api_key.clone(),
            },
        };

        let mut builder = RwkvInferRouterBuilder::new().with_state(state);
        if let Some(origins) = infer_cfg.allowed_origins.clone() {
            builder = builder.with_allowed_origins(origins);
        }
        let app = builder.build().await.expect("build router");

        let listener = tokio::net::TcpListener::bind(backend_cfg.http_bind_addr)
            .await
            .expect("bind http addr");
        axum::serve(listener, app).await.expect("serve");
    }

    #[cfg(feature = "wgpu")]
    mod wgpu {
        use super::{ElemType, launch};
        use rwkv::custom::backend::Wgpu;

        pub async fn run() {
            launch::<Wgpu<ElemType, i32>>().await;
        }
    }

    #[cfg(feature = "vulkan")]
    mod vulkan {
        use super::{ElemType, launch};
        use rwkv::custom::backend::Vulkan;

        pub async fn run() {
            launch::<Vulkan<ElemType, i32>>().await;
        }
    }

    #[cfg(feature = "metal")]
    mod metal {
        use super::{ElemType, launch};
        use rwkv::custom::backend::Metal;

        pub async fn run() {
            launch::<Metal<ElemType, i32>>().await;
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::{ElemType, launch};
        use rwkv::custom::backend::Cuda;

        pub async fn run() {
            launch::<Cuda<ElemType, i32>>().await;
        }
    }

    #[cfg(feature = "rocm")]
    mod rocm {
        use super::{ElemType, launch};
        use rwkv::custom::backend::Rocm;

        pub async fn run() {
            launch::<Rocm<ElemType, i32>>().await;
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
