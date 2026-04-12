use std::{
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fs4::fs_std::FileExt;
use rwkv_eval::cores::datasets::{ALL_BENCHMARKS, Benchmark, BenchmarkInfo};

pub mod coding;
pub mod function_calling;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;

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

fn benchmark_info(name: &str) -> &'static BenchmarkInfo {
    ALL_BENCHMARKS
        .iter()
        .find(|info| info.name.0 == name)
        .unwrap_or_else(|| panic!("unknown benchmark {name}"))
}

fn pick_sample_index(len: usize) -> usize {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as usize;
    nanos % len
}

fn rwkv_lm_eval_datasets_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("rwkv-lm-eval")
        .join("datasets")
}

fn create_benchmark(info: &BenchmarkInfo, dataset_root: &Path) -> Box<dyn Benchmark> {
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

pub(crate) async fn assert_download_dataset(name: &str) {
    let info = benchmark_info(name);
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

pub(crate) async fn assert_load_dataset(name: &str) {
    let info = benchmark_info(name);
    assert_eq!(
        info.avg_ks.len(),
        1,
        "benchmark {} should expose exactly one avg_k in tests",
        info.name.0
    );
    let avg_k = info.avg_ks[0];

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
    println!(
        "benchmark_load_summary name={} len={} avg_k={} scaled_len={} dataset_root={}",
        info.name.0,
        len,
        avg_k,
        scaled_len,
        dataset_root.display()
    );
}

pub(crate) async fn assert_show_expected_context(name: &str) {
    let info = benchmark_info(name);
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
