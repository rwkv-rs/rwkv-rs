use std::path::{Path, PathBuf};

use downloader::{UrlDownloadFile, download_url_files};
use viewer::get_parquet_files;

pub mod downloader;
pub mod viewer;

pub async fn download_hf_parquet_splits<P: AsRef<Path>>(
    path: P,
    root_name: &str,
    dataset: &str,
    config: &str,
    splits: &[&str],
    tasks: usize,
) -> PathBuf {
    let files = get_parquet_files(dataset)
        .await
        .into_iter()
        .filter(|file| file.config == config && splits.contains(&file.split.as_str()))
        .map(|file| UrlDownloadFile {
            relative_path: file.relative_path(),
            url: file.url,
        })
        .collect::<Vec<_>>();

    assert!(
        !files.is_empty(),
        "no parquet files found for dataset={dataset}, config={config}, splits={splits:?}",
    );

    download_url_files(path, root_name, &files, tasks).await
}
