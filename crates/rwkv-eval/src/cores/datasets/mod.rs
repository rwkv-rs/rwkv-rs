//! Benchmark registry and shared dataset test conventions.
//!
//! # Benchmark dataset tests
//!
//! Every benchmark module should define its local `tests` module with
//! [`benchmark_dataset_tests!`]. The macro expands to three ignored tests with
//! distinct responsibilities:
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
//! cargo nextest run -p rwkv-eval --lib --run-ignored only download_dataset --no-capture
//! cargo nextest run -p rwkv-eval --lib --run-ignored only load_dataset --no-capture
//! cargo nextest run -p rwkv-eval --lib --run-ignored only show_expected_context --no-capture
//! cargo nextest run -p rwkv-eval --lib 'cores::datasets::maths::gsm8k::tests::' --run-ignored only --no-capture
//! cargo nextest run -p rwkv-eval --lib --run-ignored only 'tau_bench::benchmark::tests::show_expected_context' --no-capture
//! ```
//!
//! When multi-file downloads are the bottleneck, increase the downloader
//! parallelism for that run:
//!
//! ```text
//! RWKV_EVAL_DOWNLOAD_TASKS=8 cargo nextest run -p rwkv-eval --lib --run-ignored only download_dataset --no-capture
//! ```
//!
//! The fourth command selects one benchmark's whole test module and lets
//! nextest schedule the three phases in order. The fifth command is the
//! shortest correct way to run just one benchmark's `show_expected_context`
//! test.
//!
//! Avoid combining positional filters like
//! `... 'tau_bench::benchmark::tests::' ... show_expected_context ...`:
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

use crate::cores::inferers::{CompletionRequest, CompletionResponse, create_completion_streamed};

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
        sandbox_queue: &crate::cores::sandbox_queue::SandboxQueue,
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

    let resp: CompletionResponse = create_completion_streamed(model_client, &req)
        .await
        .unwrap();

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

/// Shared helpers for benchmark dataset tests.
///
/// The helpers deliberately target `examples/rwkv-lm-eval/datasets` so that
/// benchmark downloads are persisted across runs and match the path used by the
/// local evaluator example.
#[cfg(test)]
pub(crate) mod test_utils {
    use std::{
        fs::{self, File, OpenOptions},
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use fs4::fs_std::FileExt;

    use super::BenchmarkInfo;

    struct SharedBenchmarkLock {
        benchmark_name: &'static str,
        lock_path: PathBuf,
        file: File,
    }

    impl Drop for SharedBenchmarkLock {
        fn drop(&mut self) {
            self.file.unlock().unwrap_or_else(|err| {
                panic!(
                    "failed to unlock shared dataset lock for {} at {}: {err}",
                    self.benchmark_name,
                    self.lock_path.display()
                )
            });
        }
    }

    fn pick_sample_index(len: usize) -> usize {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as usize;
        nanos % len
    }

    fn rwkv_lm_eval_datasets_path() -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(Path::parent)
            .unwrap_or_else(|| {
                panic!(
                    "failed to locate workspace root from {}",
                    manifest_dir.display()
                )
            });
        workspace_root
            .join("examples")
            .join("rwkv-lm-eval")
            .join("datasets")
    }

    fn create_benchmark(info: &BenchmarkInfo, dataset_root: &Path) -> Box<dyn super::Benchmark> {
        (info.create)(dataset_root.to_path_buf())
    }

    fn acquire_shared_benchmark_lock(
        info: &BenchmarkInfo,
        dataset_root: &Path,
    ) -> SharedBenchmarkLock {
        let lock_dir = dataset_root.join(".locks");
        fs::create_dir_all(&lock_dir).unwrap_or_else(|err| {
            panic!(
                "failed to create shared dataset lock dir {}: {err}",
                lock_dir.display()
            )
        });

        let lock_path = lock_dir.join(format!("{}.lock", info.name.0));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to open shared dataset lock {}: {err}",
                    lock_path.display()
                )
            });

        file.lock_exclusive().unwrap_or_else(|err| {
            panic!(
                "failed to lock shared dataset {} at {}: {err}",
                info.name.0,
                lock_path.display()
            )
        });

        SharedBenchmarkLock {
            benchmark_name: info.name.0,
            lock_path,
            file,
        }
    }

    pub(crate) async fn assert_download_dataset(info: &BenchmarkInfo) {
        let dataset_root = rwkv_lm_eval_datasets_path();
        fs::create_dir_all(&dataset_root).unwrap_or_else(|err| {
            panic!(
                "failed to create shared dataset root {}: {err}",
                dataset_root.display()
            )
        });
        let _lock = acquire_shared_benchmark_lock(info, &dataset_root);

        let mut benchmark = create_benchmark(info, &dataset_root);
        if benchmark.load() {
            println!(
                "benchmark_download_start name={} dataset_root={}",
                info.name.0,
                dataset_root.display()
            );
            benchmark.download().await;
        } else {
            println!(
                "benchmark_download_reused name={} dataset_root={}",
                info.name.0,
                dataset_root.display()
            );
        }

        let mut reloaded = create_benchmark(info, &dataset_root);
        assert!(
            !reloaded.load(),
            "benchmark {} failed to load after download/reuse from {}",
            info.name.0,
            dataset_root.display()
        );
        println!(
            "benchmark_download_ok name={} dataset_root={}",
            info.name.0,
            dataset_root.display()
        );
    }

    pub(crate) async fn assert_load_dataset(info: &BenchmarkInfo) {
        assert!(
            !info.avg_ks.is_empty(),
            "benchmark {} should expose at least one avg_k",
            info.name.0
        );
        for &avg_k in info.avg_ks {
            assert!(
                avg_k.is_finite() && avg_k > 0.0 && avg_k <= 64.0,
                "benchmark {} has invalid avg_k={avg_k}; avg_k must be finite, > 0, and <= 64",
                info.name.0
            );
            if avg_k >= 1.0 {
                let rounded = avg_k.round();
                assert!(
                    (avg_k - rounded).abs() <= f32::EPSILON,
                    "benchmark {} has invalid avg_k={avg_k}; avg_k >= 1 must be an integer power of two <= 64",
                    info.name.0
                );
                let integer = rounded as u32;
                assert!(
                    integer.is_power_of_two(),
                    "benchmark {} has invalid avg_k={avg_k}; avg_k >= 1 must be an integer power of two <= 64",
                    info.name.0
                );
            }
        }
        let avg_k = info.avg_ks.iter().copied().max_by(f32::total_cmp).unwrap();

        let dataset_root = rwkv_lm_eval_datasets_path();
        let _lock = acquire_shared_benchmark_lock(info, &dataset_root);
        let mut benchmark = create_benchmark(info, &dataset_root);

        assert!(
            !benchmark.load(),
            "benchmark {} failed to load from {}. run download_dataset first",
            info.name.0,
            dataset_root.display()
        );

        let len = benchmark.len();
        assert!(len > 0, "benchmark {} loaded zero samples", info.name.0);

        let scaled_len = len as f32 * avg_k;
        if avg_k < 64.0 {
            assert!(
                scaled_len >= 4000.0,
                "benchmark {} expected len * max(avg_ks) >= 4000, got len={} max_avg_k={} scaled_len={} dataset_root={}",
                info.name.0,
                len,
                avg_k,
                scaled_len,
                dataset_root.display()
            );
        }
        println!(
            "benchmark_load_summary name={} len={} avg_k={} scaled_len={} dataset_root={}",
            info.name.0,
            len,
            avg_k,
            scaled_len,
            dataset_root.display()
        );
    }

    pub(crate) async fn assert_show_expected_context(info: &BenchmarkInfo) {
        let cot_mode = *info
            .cot_mode
            .first()
            .unwrap_or_else(|| panic!("benchmark {} missing supported cot_mode", info.name.0));
        let n_shot = *info
            .n_shots
            .first()
            .unwrap_or_else(|| panic!("benchmark {} missing supported n_shot", info.name.0));
        let dataset_root = rwkv_lm_eval_datasets_path();
        let _lock = acquire_shared_benchmark_lock(info, &dataset_root);
        let mut benchmark = create_benchmark(info, &dataset_root);
        assert!(
            !benchmark.load(),
            "benchmark {} failed to load from {}. run download_dataset first",
            info.name.0,
            dataset_root.display()
        );

        let len = benchmark.len();
        assert!(len > 0, "benchmark {} loaded zero samples", info.name.0);
        let sample_index = pick_sample_index(len);
        let expected_context = benchmark.get_expected_context(sample_index, cot_mode, n_shot);
        assert!(
            !expected_context.trim().is_empty(),
            "benchmark {} produced empty expected context at sample index {sample_index}",
            info.name.0
        );

        println!(
            concat!(
                "\n==================== Benchmark Sample ====================\n",
                "name: {}\n",
                "display_name: {}\n",
                "sample_index: {}\n",
                "len: {}\n",
                "cot_mode: {:?}\n",
                "n_shot: {}\n",
                "dataset_root: {}\n",
                "-------------------- expected_context --------------------\n",
                "{}\n",
                "==========================================================\n"
            ),
            info.name.0,
            info.display_name,
            sample_index,
            len,
            cot_mode,
            n_shot,
            dataset_root.display(),
            expected_context
        );
    }
}

/// Generate the standard three-phase dataset tests for a benchmark.
///
/// The expanded tests are intentionally marked `#[ignore]` because they may
/// touch the network or depend on previously downloaded shared datasets. Run
/// them with `cargo nextest --run-ignored only ...` so the nextest config can
/// prioritize the three phases while the test helpers serialize access to each
/// benchmark's shared dataset cache entry.
#[cfg(test)]
macro_rules! benchmark_dataset_tests {
    ($info:expr) => {
        #[tokio::test]
        #[ignore = "downloads remote benchmark dataset into examples/rwkv-lm-eval/datasets"]
        async fn download_dataset() {
            $crate::cores::datasets::test_utils::assert_download_dataset(&$info).await;
        }

        #[tokio::test]
        #[ignore = "loads benchmark dataset from examples/rwkv-lm-eval/datasets"]
        async fn load_dataset() {
            $crate::cores::datasets::test_utils::assert_load_dataset(&$info).await;
        }

        #[tokio::test]
        #[ignore = "renders benchmark context from examples/rwkv-lm-eval/datasets"]
        async fn show_expected_context() {
            $crate::cores::datasets::test_utils::assert_show_expected_context(&$info).await;
        }
    };
}

#[cfg(test)]
pub(crate) use benchmark_dataset_tests;
