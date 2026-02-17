//! End-to-end single-prompt continuation test via the in-process SDK.
//!
//! This is intentionally `#[ignore]` because it requires:
//! - a local `.mpk` weights file (large);
//! - a working CUDA setup (by default).
//!
//! Run (example):
//!   RWKV_TEST_MPK=examples/rwkv-lm/weights/rwkv7-g1d-7.2b-20260131-ctx8192.mpk \
//!   RWKV_TEST_VOCAB=examples/rwkv-lm/assets/rwkv_vocab_v20230424.txt \
//!   cargo test -p rwkv-lm --test sdk_single_prompt -- --ignored --nocapture
//!
//! Optional assertions:
//! - RWKV_TEST_EXPECT_SUBSTR="hello"  (case-insensitive substring check)

#![cfg(all(feature = "inferring", feature = "cuda"))]

use std::sync::Arc;

use rwkv::custom::backend::Cuda;
use rwkv::custom::cubecl::device::DeviceId;
use rwkv::custom::module::Module;
use rwkv::custom::prelude::{Backend, DeviceOps};
use rwkv::custom::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rwkv::custom::tensor::bf16;

use rwkv::infer::engine::{EngineRuntime, EngineRuntimeConfig};
use rwkv::infer::sdk::RwkvInferClient;
use rwkv::infer::types::EngineEvent;
use rwkv::infer::SamplingConfig;

use rwkv_lm::inferring::RwkvLmExecutor;
use rwkv_lm::model::AutoRegressiveModelConfig;

type MyBackend = Cuda<bf16, i32>;

fn env_opt(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn sdk_single_prompt_continuation_smoke() {
    // --- config ---
    let mpk_path = env_opt("RWKV_TEST_MPK")
        .unwrap_or_else(|| "examples/rwkv-lm/weights/model.mpk".to_string());
    let vocab_path = env_opt("RWKV_TEST_VOCAB")
        .unwrap_or_else(|| "examples/rwkv-lm/assets/rwkv_vocab_v20230424.txt".to_string());

    if !std::path::Path::new(&mpk_path).exists() {
        eprintln!("skip: RWKV_TEST_MPK not found: {mpk_path}");
        return;
    }
    if !std::path::Path::new(&vocab_path).exists() {
        eprintln!("skip: RWKV_TEST_VOCAB not found: {vocab_path}");
        return;
    }

    // This test targets the common RWKV-7 7B config; override by editing if needed.
    let num_cells = 32;
    let vocab_size = 65536;
    let embedded_dim = 4096;
    let num_heads = 64;
    let head_size = 64;

    // --- build engine runtime (in-process) ---
    let device = <MyBackend as Backend>::Device::from_id(DeviceId::new(0, 0));

    let model_config =
        AutoRegressiveModelConfig::new(num_cells, vocab_size, embedded_dim, num_heads, head_size);
    let model = model_config.init::<MyBackend>(&device);
    let model = model
        .load_file(
            &mpk_path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .expect("load mpk weights");

    let executor = RwkvLmExecutor::<MyBackend>::new(
        device.clone(),
        model,
        /* max_batch_size */ 1,
        num_cells,
        vocab_size,
        embedded_dim,
        num_heads,
        head_size,
    );

    let engine = EngineRuntime::spawn(
        EngineRuntimeConfig {
            tokenizer_vocab_path: vocab_path,
            max_batch_size: 1,
            paragraph_len: 256,
            max_context_len: 8192,
            decode_first: true,
        },
        Box::new(executor),
    )
    .expect("spawn engine runtime");

    let client = RwkvInferClient::new(Arc::new(engine));

    // --- run a single continuation ---
    // Keep the prompt in the well-known RWKV chat format (no special <think> prefix here).
    let prompt = "User: Hello!\n\nAssistant:".to_string();
    let sampling = SamplingConfig {
        // Deterministic argmax path (matches the CUDA reference behavior):
        // top_p==0 => kernel normalizes to top_k=1, top_p=1.
        temperature: 1.0,
        top_k: 0,
        top_p: 0.0,
        max_new_tokens: 64,
        presence_penalty: 0.0,
        repetition_penalty: 0.0,
        penalty_decay: 1.0,
    };

    let out = client
        .completions_text(prompt, sampling, vec![], /* stream */ true)
        .await
        .expect("sdk submit");

    let mut rx = match out {
        rwkv::infer::engine::SubmitOutput::Stream { rx, .. } => rx,
        other => panic!("expected streaming output, got {other:?}"),
    };

    let mut text_out = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            EngineEvent::Text(t) => text_out.push_str(&t),
            EngineEvent::Done => break,
            EngineEvent::Error(e) => panic!("engine error: {e}"),
        }
    }

    eprintln!("=== continuation ===\n{text_out}\n====================");

    assert!(!text_out.trim().is_empty(), "empty continuation");
    assert!(
        !text_out.contains('\0'),
        "continuation contains NUL byte"
    );

    if let Some(expect) = env_opt("RWKV_TEST_EXPECT_SUBSTR") {
        let hay = text_out.to_lowercase();
        let needle = expect.to_lowercase();
        assert!(
            hay.contains(&needle),
            "continuation does not contain expected substring: {expect:?}"
        );
    }
}
