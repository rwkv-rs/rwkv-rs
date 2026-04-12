//! Benchmark registry and shared dataset test conventions.
//!
//! # Benchmark dataset tests
//!
//! Every benchmark module is covered by a mirrored integration test file under
//! `crates/rwkv-eval/tests/cores/datasets/**`. The shared helpers generate
//! three ignored tests with distinct responsibilities:
//!
//! 1. `download_dataset`
//! 2. `load_dataset`
//! 3. `show_expected_context`
//!
//! These tests intentionally share the dataset cache under
//! `examples/rwkv-lm-eval/datasets` instead of creating temporary directories.
//! The first run performs a real download when the benchmark cannot be loaded
//! from disk; subsequent runs reuse the existing files.
//!
//! `load_dataset` and `show_expected_context` assume the shared data
//! is already present. If the dataset has not been downloaded yet, they fail
//! with a message telling the caller to run the download test first.
//!
//! # How to run
//!
//! Use `cargo nextest`, not plain `cargo test`, for these ignored dataset
//! tests. The repository-level nextest config gives them the expected
//! `download -> load -> render` priority when a command matches more than one
//! phase, and the test helpers take a per-benchmark file lock so different
//! benchmarks can still run in parallel against the shared cache.
//!
//! Recommended commands:
//!
//! ```text
//! cargo nextest run -p rwkv-eval --test benchmark_datasets --run-ignored only download_dataset --no-capture
//! cargo nextest run -p rwkv-eval --test benchmark_datasets --run-ignored only load_dataset --no-capture
//! cargo nextest run -p rwkv-eval --test benchmark_datasets --run-ignored only show_expected_context --no-capture
//! cargo nextest run -p rwkv-eval --test benchmark_datasets 'cores::datasets::maths::gsm8k::' --run-ignored only --no-capture
//! cargo nextest run -p rwkv-eval --test benchmark_datasets --run-ignored only 'tau_bench::benchmark::show_expected_context' --no-capture
//! ```
//!
//! When multi-file downloads are the bottleneck, increase the downloader
//! parallelism for that run:
//!
//! ```text
//! RWKV_EVAL_DOWNLOAD_TASKS=8 cargo nextest run -p rwkv-eval --test benchmark_datasets --run-ignored only download_dataset --no-capture
//! ```
//!
//! The fourth command selects one benchmark's whole test module and lets
//! nextest schedule the three phases in order. The fifth command is the
//! shortest correct way to run just one benchmark's `show_expected_context`
//! test.
//!
//! Avoid combining positional filters like
//! `... 'tau_bench::benchmark::' ... show_expected_context ...`:
//! nextest treats them as a union, so the trailing `show_expected_context`
//! matches every benchmark's test with that name.
//!
pub mod coding;
pub mod function_calling;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;
pub mod utils;

use std::{collections::BTreeMap, path::PathBuf};

use async_openai::{Client, config::OpenAIConfig};
use async_trait::async_trait;
use linkme::distributed_slice;
use once_cell::sync::Lazy;

use crate::cores::{
    inferers::{CompletionRequest, CompletionResponse, create_completion_streamed},
    sandbox_queue::SandboxQueue,
};

pub struct BenchmarkInfo {
    pub name: BenchmarkName,
    pub field: Field,
    pub display_name: &'static str,
    pub cot_mode: &'static [CoTMode],
    pub sampling_config: SamplingConfig,
    pub n_shots: &'static [u8],
    pub avg_ks: &'static [f32],
    pub pass_ks: &'static [u8],
    pub with_llm_judger: bool,
    pub create: BenchmarkFactory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BenchmarkName(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Knowledge,
    Maths,
    Coding,
    InstructionFollowing,
    FunctionCalling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoTMode {
    NoCoT,
    FakeCoT,
    CoT,
}

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,

    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub penalty_decay: f32,
}

#[derive(Debug, Clone)]
pub struct Record {
    pub context: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
}

pub type BenchmarkFactory = fn(PathBuf) -> Box<dyn Benchmark>;

#[async_trait]
pub trait Benchmark: Send + Sync {
    fn load(&mut self) -> bool; // return is_invalid
    async fn check(&self) -> bool; // return is_invalid
    async fn download(&self);
    fn len(&self) -> usize;

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String;
    fn get_ref_answer(&self, index: usize) -> String;
    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        judger_model_name: Option<&str>,
        judger_client: Option<&Client<OpenAIConfig>>,
        sandbox_queue: &SandboxQueue,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record;
}

#[distributed_slice]
pub static ALL_BENCHMARKS: [BenchmarkInfo] = [..];

pub static BENCHMARKS_BY_FIELD: Lazy<BTreeMap<Field, Vec<&'static BenchmarkInfo>>> =
    Lazy::new(|| {
        let mut map: BTreeMap<Field, Vec<&'static BenchmarkInfo>> = BTreeMap::new();

        for info in ALL_BENCHMARKS {
            map.entry(info.field).or_default().push(info);
        }

        // 两百个 benchmark 后，顺序不要赌链接器/注册顺序，统一显式排序
        for vec_info in map.values_mut() {
            vec_info.sort_unstable_by_key(|m| m.name.0);
        }

        map
    });

pub fn get_benchmarks_with_field(field: Field) -> &'static [&'static BenchmarkInfo] {
    static EMPTY: &[&BenchmarkInfo] = &[];
    BENCHMARKS_BY_FIELD
        .get(&field)
        .map(Vec::as_slice)
        .unwrap_or(EMPTY)
}

pub fn apply_user_assistant_template(user_part: String, assistant_part: String) -> String {
    format!("User: {user_part}\n\nAssistant: {assistant_part}")
}

pub fn get_prompt_for_cot(expected_context: &str) -> String {
    expected_context
        .split_once("<|completions_of_cot|>")
        .unwrap()
        .0
        .to_string()
}

pub async fn get_completions_of_cot(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt_for_cot: &str,
    sampling_config: &SamplingConfig,
) -> String {
    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt_for_cot.into(),
        vec!["</think>".to_string()],
        4096,
        &sampling_config,
        None,
        None,
    );

    let resp: CompletionResponse = create_completion_streamed(model_client, &req).await.unwrap();

    resp.choices[0].text.clone()
}

pub fn get_prompt_for_final_answer(
    expected_context: &str,
    completions_of_cot: Option<&str>,
) -> String {
    completions_of_cot
        .map(|cot| expected_context.replace("<|completions_of_cot|>", cot))
        .unwrap_or_else(|| expected_context.to_string())
        .split_once("<|logprobs_of_choices|>")
        .unwrap()
        .0
        .to_string()
}

pub fn render_context(expected_context: &str, replacements: &[(&str, &str)]) -> String {
    let mut context = expected_context.to_string();
    for (placeholder, value) in replacements {
        context = context.replace(placeholder, value);
    }
    context
}
