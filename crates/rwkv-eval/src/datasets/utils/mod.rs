use std::path::{Path, PathBuf};

use walkdir::WalkDir;

pub mod csv;
pub mod hf;
pub mod jsonl;
pub mod parquet;

pub fn collect_files_with_extension<P: AsRef<Path>>(root: P, extension: &str) -> Vec<PathBuf> {
    let mut files = WalkDir::new(root.as_ref())
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    files.sort();
    files
}
