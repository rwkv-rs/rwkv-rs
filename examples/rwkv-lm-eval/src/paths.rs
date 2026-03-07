use std::path::{Path, PathBuf};

pub fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

pub fn config_dir() -> PathBuf {
    crate_root().join("config")
}

pub fn datasets_dir() -> PathBuf {
    crate_root().join("datasets")
}
