use std::{
    borrow::Cow,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
};

use tokio::sync::mpsc::Receiver;

pub mod reader;
pub mod writer;

#[derive(Debug, Clone)]

pub enum DataItem {
    FileStart(String), // 相对路径，用下划线连接，不包含后缀
    DataBatch(Vec<Cow<'static, str>>),
}

pub trait Reader {
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>>;
}

pub trait Writer {
    fn run(&self, rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

pub fn find_common_prefix(paths: &[PathBuf]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }

    let first = paths[0].clone();

    let mut common = PathBuf::new();

    for component in first.components() {
        let candidate = common.join(component);

        if paths.iter().all(|p| p.starts_with(&candidate)) {
            common = candidate;
        } else {
            break;
        }
    }

    common
}

pub fn generate_relative_name(input_path: &Path, common_prefix: &Path) -> String {
    let relative = input_path.strip_prefix(common_prefix).unwrap_or(input_path);

    let stem = relative.with_extension(""); // 移除后缀
    stem.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("_")
}
