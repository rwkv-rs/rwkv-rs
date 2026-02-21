use std::path::{Path, PathBuf};

pub fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

pub fn logs_dir() -> PathBuf {
    crate_root().join("logs")
}

pub fn bench_logs_dir() -> PathBuf {
    logs_dir().join("bench")
}

pub fn bench_output_path(name: &str) -> PathBuf {
    bench_logs_dir().join(name)
}
