use std::path::{Path, PathBuf};

pub fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

pub fn config_dir() -> PathBuf {
    crate_root().join("config")
}

pub fn data_dir() -> PathBuf {
    crate_root().join("data")
}

pub fn default_config_path() -> PathBuf {
    config_dir().join("example.toml")
}
