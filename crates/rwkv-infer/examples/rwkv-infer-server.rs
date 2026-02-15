use std::sync::Arc;

use rwkv_config::{
    load_toml,
    raw::infer::RawInferConfig,
    validated::infer::{FinalInferConfig, FinalInferConfigBuilder},
};
use rwkv_infer::auth::AuthConfig;
use rwkv_infer::config::BackendConfig;
use rwkv_infer::engine::{EngineRuntime, EngineRuntimeConfig, InferExecutor};
use rwkv_infer::server::{RwkvInferRouterBuilder, SharedRwkvInferState};

#[cfg(feature = "tokenizer")]
use rwkv_data::tokenizer::Tokenizer;

struct DummyExecutor {
    #[cfg(feature = "tokenizer")]
    tokenizer: Tokenizer,
    token_a: i32,
}

impl DummyExecutor {
    #[cfg(feature = "tokenizer")]
    fn new(vocab_path: &str) -> Self {
        let tokenizer = Tokenizer::new(vocab_path).expect("load vocab");
        let token_a = tokenizer.encode("a", false).get(0).copied().unwrap_or(0) as i32;
        Self { tokenizer, token_a }
    }
}

impl InferExecutor for DummyExecutor {
    fn tokenize(&self, text: &str) -> rwkv_infer::Result<Vec<i32>> {
        #[cfg(feature = "tokenizer")]
        {
            Ok(self
                .tokenizer
                .encode(text, false)
                .into_iter()
                .map(|t| t as i32)
                .collect())
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = text;
            Err(rwkv_infer::Error::NotSupported(
                "feature tokenizer is disabled",
            ))
        }
    }

    fn detokenize(&self, token_ids: &[i32]) -> rwkv_infer::Result<String> {
        #[cfg(feature = "tokenizer")]
        {
            Ok(self.tokenizer.decode(
                token_ids
                    .iter()
                    .copied()
                    .map(|t| t as u16)
                    .collect::<Vec<_>>(),
            ))
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = token_ids;
            Err(rwkv_infer::Error::NotSupported(
                "feature tokenizer is disabled",
            ))
        }
    }

    fn prefill(&mut self, _batch_positions: &[(usize, &[i32], &[u8])]) -> rwkv_infer::Result<()> {
        Ok(())
    }

    fn decode(
        &mut self,
        batch_positions: &[(usize, i32)],
        _sampling: rwkv_infer::SamplingConfig,
    ) -> rwkv_infer::Result<Vec<(usize, i32)>> {
        Ok(batch_positions
            .iter()
            .map(|(batch_index, _)| (*batch_index, self.token_a))
            .collect())
    }

    fn reset_batch_position(&mut self, _batch_index: usize) -> rwkv_infer::Result<()> {
        Ok(())
    }
}

fn load_infer_config(path: &str) -> Arc<FinalInferConfig> {
    let mut raw: RawInferConfig = load_toml(path);
    raw.fill_default();
    let builder = FinalInferConfigBuilder::load_from_raw(raw);
    builder.build()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let infer_cfg = load_infer_config("examples/rwkv-lm/config/infer.toml");
    let backend_cfg = BackendConfig::from(infer_cfg.as_ref());

    let executor = DummyExecutor::new(&infer_cfg.tokenizer_vocab_path);
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
    let app = builder.build().await?;
    let listener = tokio::net::TcpListener::bind(backend_cfg.http_bind_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
