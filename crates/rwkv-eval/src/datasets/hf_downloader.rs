use std::fs::{File, create_dir_all, remove_dir_all};
use std::future::Future;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use tokio::sync::Semaphore;
use tokio::time::sleep;
use walkdir::WalkDir;

pub async fn download_hf_repo<P: AsRef<Path>>(
    path: P,
    repo: &str,
    tasks: usize,
    revision: &str,
) -> PathBuf {
    assert_ne!(tasks, 0, "tasks 不能为 0");

    let path = path.as_ref();
    create_dir_all(path).unwrap_or_else(|e| {
        panic!("创建目标目录失败: {}. error: {}", path.display(), e);
    });

    // 规范化 repo（得到：url、repo_id(可能含 datasets/ 或 spaces/)、repo_name）
    let (url, repo_id, repo_name) = normalize_repo(repo);
    let repo_dir = path.join(&repo_name);

    // 1) git clone（不拉 LFS）
    clone_repo_with_retry(&url, &repo_dir).await;

    // 2) 扫描 LFS pointer
    let lfs = scan_lfs_pointers(&repo_dir);
    if lfs.is_empty() {
        // 没有 LFS，大概率已经完整；直接返回
        return repo_dir;
    }

    // 3) 进度条：按总字节
    let total_bytes: u64 = lfs.iter().map(|x| x.size).sum();
    let pb = ProgressBar::new(total_bytes);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg}\n{bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(format!("Downloading LFS files ({} files)", lfs.len()));

    // 4) 并发下载并覆盖 pointer 文件
    let base = "https://hf-mirror.com".trim_end_matches('/').to_string();
    let client = Client::new();
    let sem = Arc::new(Semaphore::new(tasks));

    let mut handles = Vec::with_capacity(lfs.len());
    for item in lfs {
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .unwrap_or_else(|e| panic!("获取下载并发许可失败: {}", e));
        let client = client.clone();
        let pb = pb.clone();
        let repo_dir = repo_dir.clone();
        let base = base.clone();
        let repo_id = repo_id.clone();
        let revision = revision.to_string();

        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let url = format!(
                "{}/{}/resolve/{}/{}",
                base,
                repo_id,
                revision,
                item.rel_path.to_string_lossy()
            );
            let out_path = repo_dir.join(&item.rel_path);

            let body = retry(
                &format!("下载 LFS 文件 {}", item.rel_path.display()),
                || {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let resp = client
                            .get(&url)
                            .send()
                            .await
                            .map_err(|e| format!("请求发送失败: {}", e))?;
                        let status = resp.status();
                        if !status.is_success() {
                            return Err(format!("HTTP 状态码异常: {}", status));
                        }
                        resp.bytes()
                            .await
                            .map_err(|e| format!("读取响应体失败: {}", e))
                    }
                },
            )
            .await;

            if let Some(parent) = out_path.parent() {
                create_dir_all(parent).unwrap_or_else(|e| {
                    panic!("创建文件父目录失败: {}. error: {}", parent.display(), e);
                });
            }

            let mut file = File::create(&out_path).unwrap_or_else(|e| {
                panic!("创建输出文件失败: {}. error: {}", out_path.display(), e);
            });
            file.write_all(&body).unwrap_or_else(|e| {
                panic!("写入文件失败: {}. error: {}", out_path.display(), e);
            });
            file.flush().unwrap_or_else(|e| {
                panic!("刷新文件失败: {}. error: {}", out_path.display(), e);
            });
            pb.inc(body.len() as u64);
        }));
    }

    for h in handles {
        h.await
            .unwrap_or_else(|e| panic!("下载任务 join 失败: {}", e));
    }

    pb.finish_with_message("Done");
    repo_dir
}

pub async fn download_hf_files<P: AsRef<Path>>(
    path: P,
    repo: &str,
    files: &[&str],
    tasks: usize,
    revision: &str,
) -> PathBuf {
    assert_ne!(tasks, 0, "tasks 不能为 0");
    assert!(!files.is_empty(), "files 不能为空");

    let path = path.as_ref();
    create_dir_all(path).unwrap_or_else(|e| {
        panic!("创建目标目录失败: {}. error: {}", path.display(), e);
    });

    let (_, repo_id, repo_name) = normalize_repo(repo);
    let repo_dir = path.join(&repo_name);
    if repo_dir.exists() {
        remove_dir_all(&repo_dir).unwrap_or_else(|e| {
            panic!("删除已有仓库目录失败: {}. error: {}", repo_dir.display(), e);
        });
    }
    create_dir_all(&repo_dir).unwrap_or_else(|e| {
        panic!("创建仓库目录失败: {}. error: {}", repo_dir.display(), e);
    });

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{msg}\n{bar:40.cyan/blue} {pos}/{len} ({per_sec}, {eta})")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(format!(
        "Downloading selected files ({} files)",
        files.len()
    ));

    let base = "https://hf-mirror.com".trim_end_matches('/').to_string();
    let client = Client::new();
    let sem = Arc::new(Semaphore::new(tasks));

    let mut handles = Vec::with_capacity(files.len());
    for rel_path in files {
        let permit = sem
            .clone()
            .acquire_owned()
            .await
            .unwrap_or_else(|e| panic!("获取下载并发许可失败: {}", e));
        let client = client.clone();
        let pb = pb.clone();
        let repo_dir = repo_dir.clone();
        let base = base.clone();
        let repo_id = repo_id.clone();
        let revision = revision.to_string();
        let rel_path = (*rel_path).to_string();

        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let url = format!("{}/{}/resolve/{}/{}", base, repo_id, revision, rel_path);
            let out_path = repo_dir.join(&rel_path);

            let body = retry(&format!("下载文件 {}", rel_path), || {
                let client = client.clone();
                let url = url.clone();
                async move {
                    let resp = client
                        .get(&url)
                        .send()
                        .await
                        .map_err(|e| format!("请求发送失败: {}", e))?;
                    let status = resp.status();
                    if !status.is_success() {
                        return Err(format!("HTTP 状态码异常: {}", status));
                    }
                    resp.bytes()
                        .await
                        .map_err(|e| format!("读取响应体失败: {}", e))
                }
            })
            .await;

            if let Some(parent) = out_path.parent() {
                create_dir_all(parent).unwrap_or_else(|e| {
                    panic!("创建文件父目录失败: {}. error: {}", parent.display(), e);
                });
            }

            let mut file = File::create(&out_path).unwrap_or_else(|e| {
                panic!("创建输出文件失败: {}. error: {}", out_path.display(), e);
            });
            file.write_all(&body).unwrap_or_else(|e| {
                panic!("写入文件失败: {}. error: {}", out_path.display(), e);
            });
            file.flush().unwrap_or_else(|e| {
                panic!("刷新文件失败: {}. error: {}", out_path.display(), e);
            });
            pb.inc(1);
        }));
    }

    for h in handles {
        h.await
            .unwrap_or_else(|e| panic!("下载任务 join 失败: {}", e));
    }

    pb.finish_with_message("Done");
    repo_dir
}

async fn clone_repo_with_retry(url: &str, repo_dir: &Path) {
    retry("git clone", || {
        let url = url.to_string();
        let repo_dir = repo_dir.to_path_buf();
        async move {
            if repo_dir.exists() {
                remove_dir_all(&repo_dir).unwrap_or_else(|e| {
                    panic!("删除已有仓库目录失败: {}. error: {}", repo_dir.display(), e);
                });
            }

            let status = Command::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg(&url)
                .arg(&repo_dir)
                .status()
                .map_err(|e| format!("执行 git clone 命令失败: {}", e))?;

            if !status.success() {
                return Err(format!(
                    "git clone 失败: url={}, dst={}, status={}",
                    url,
                    repo_dir.display(),
                    status
                ));
            }

            Ok(())
        }
    })
    .await;
}

async fn retry<T, F, Fut>(operation_name: &str, mut operation: F) -> T
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, String>>,
{
    let mut last_error = String::new();
    for attempt in 1..=5 {
        match operation().await {
            Ok(v) => return v,
            Err(e) => {
                last_error = e;
                if attempt < 5 {
                    let backoff_secs = 1_u64 << (attempt - 1);
                    eprintln!(
                        "[network retry {}/5] {} 失败: {}. {} 秒后重试...",
                        attempt, operation_name, last_error, backoff_secs
                    );
                    sleep(Duration::from_secs(backoff_secs)).await;
                }
            }
        }
    }

    panic!(
        "网络操作失败（已重试 5 次）: {}. 最后一次错误: {}",
        operation_name, last_error
    );
}

#[derive(Debug, Clone)]
struct LfsItem {
    rel_path: PathBuf,
    size: u64,
}

/// 扫描 repo 下所有 LFS pointer 文件，返回其相对路径 + size
fn scan_lfs_pointers(repo_dir: &Path) -> Vec<LfsItem> {
    let mut out = Vec::new();

    for entry in WalkDir::new(repo_dir) {
        let entry = entry.unwrap_or_else(|e| {
            panic!("遍历目录失败: {}. error: {}", repo_dir.display(), e);
        });

        if !entry.file_type().is_file() {
            continue;
        }

        // LFS pointer 很小：读取最多 2KB + 1 字节用于区分
        let p = entry.path();
        let mut file = File::open(p).unwrap_or_else(|e| {
            panic!("打开文件失败: {}. error: {}", p.display(), e);
        });

        let mut data = [0_u8; 2049];
        let n = file.read(&mut data).unwrap_or_else(|e| {
            panic!("读取文件失败: {}. error: {}", p.display(), e);
        });

        if n > 2048 {
            continue;
        }

        let s = match std::str::from_utf8(&data[..n]) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if !s.starts_with("version https://git-lfs.github.com/spec/v1") {
            continue;
        }

        // 找 size
        let mut size: Option<u64> = None;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("size ") {
                size = rest.trim().parse::<u64>().ok();
                break;
            }
        }
        let Some(size) = size else {
            continue;
        };

        let rel = p
            .strip_prefix(repo_dir)
            .unwrap_or_else(|e| {
                panic!(
                    "strip_prefix 失败: path={}, repo_dir={}, error={}",
                    p.display(),
                    repo_dir.display(),
                    e
                );
            })
            .to_path_buf();

        out.push(LfsItem {
            rel_path: rel,
            size,
        });
    }

    out
}

/// 统一支持：
/// - "org/repo"
/// - "datasets/org/repo"
/// - "spaces/org/repo"
/// - 或对应的完整 URL
fn normalize_repo(repo: &str) -> (String, String, String) {
    let repo = repo.trim();

    let (url, path) = if repo.starts_with("http://") || repo.starts_with("https://") {
        // 直接用给定 URL clone
        let u = repo.trim_end_matches('/').to_string();
        let without_scheme = u
            .split_once("://")
            .map(|(_, x)| x)
            .unwrap_or_else(|| panic!("URL 格式不对: {}", repo));
        let path = without_scheme
            .split_once('/')
            .map(|(_, x)| x)
            .unwrap_or("")
            .trim_end_matches(".git")
            .to_string();
        (u, path)
    } else {
        // 用官方域名 clone
        let path = repo.trim_matches('/').trim_end_matches(".git").to_string();
        let u = format!("https://huggingface.co/{}", path);
        (u, path)
    };

    // path 可能是: "datasets/cais/mmlu" 或 "Qwen/Qwen2.5-7B"
    let seg: Vec<&str> = path.split('/').filter(|x| !x.is_empty()).collect();
    if seg.len() < 2 {
        panic!("repo 格式不对: {}", repo);
    }

    let (repo_id, repo_name) = match seg[0] {
        "datasets" | "spaces" => {
            if seg.len() < 3 {
                panic!("repo 格式不对（需要 {}/<org>/<name>）: {}", seg[0], repo);
            }
            (
                format!("{}/{}/{}", seg[0], seg[1], seg[2]),
                seg[2].to_string(),
            )
        }
        _ => (format!("{}/{}", seg[0], seg[1]), seg[1].to_string()),
    };

    (url, repo_id, repo_name)
}

#[cfg(test)]
mod tests {
    use super::normalize_repo;

    #[test]
    fn normalize_standard_repo() {
        let (url, repo_id, repo_name) = normalize_repo("Qwen/Qwen2.5-7B");
        assert_eq!(url, "https://huggingface.co/Qwen/Qwen2.5-7B");
        assert_eq!(repo_id, "Qwen/Qwen2.5-7B");
        assert_eq!(repo_name, "Qwen2.5-7B");
    }

    #[test]
    fn normalize_dataset_repo() {
        let (url, repo_id, repo_name) = normalize_repo("datasets/cais/mmlu");
        assert_eq!(url, "https://huggingface.co/datasets/cais/mmlu");
        assert_eq!(repo_id, "datasets/cais/mmlu");
        assert_eq!(repo_name, "mmlu");
    }

    #[test]
    fn normalize_full_url_repo() {
        let (url, repo_id, repo_name) =
            normalize_repo("https://huggingface.co/datasets/cais/mmlu/");
        assert_eq!(url, "https://huggingface.co/datasets/cais/mmlu");
        assert_eq!(repo_id, "datasets/cais/mmlu");
        assert_eq!(repo_name, "mmlu");
    }
}
