mod diamond;
mod extended;
mod main;

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};

pub fn gpqa_csv_path<P: AsRef<Path>>(dataset_root: P, file_name: &str) -> PathBuf {
    dataset_root.as_ref().join("gpqa").join(file_name)
}

pub async fn download_gpqa_csv<P: AsRef<Path>>(dataset_root: P, file_name: &str) -> PathBuf {
    let dataset_root = dataset_root.as_ref();
    let root_dir = download_url_files(
        dataset_root,
        "gpqa",
        &[UrlDownloadFile {
            relative_path: PathBuf::from("dataset.zip"),
            url: "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip".to_string(),
        }],
        1,
    )
    .await;

    let zip_path = root_dir.join("dataset.zip");
    let archived_path = format!("dataset/{file_name}");
    let status = Command::new("unzip")
        .arg("-P")
        .arg("deserted-untie-orchid")
        .arg("-j")
        .arg("-o")
        .arg(&zip_path)
        .arg(&archived_path)
        .arg("-d")
        .arg(&root_dir)
        .status()
        .unwrap_or_else(|e| panic!("解压 GPQA 文件失败: {}. error: {}", zip_path.display(), e));

    if !status.success() {
        panic!(
            "解压 GPQA 文件失败: zip={}, entry={}, status={}",
            zip_path.display(),
            archived_path,
            status
        );
    }

    root_dir.join(file_name)
}

pub fn ordered_gpqa_choices(
    record_id: &str,
    question: &str,
    correct_answer: &str,
    incorrect_answers: [&str; 3],
) -> (Vec<String>, u8) {
    let seed_source = if record_id.trim().is_empty() {
        question
    } else {
        record_id
    };
    let rotation = seed_source.bytes().fold(0usize, |acc, byte| {
        acc.wrapping_mul(131).wrapping_add(byte as usize)
    }) % 4;

    let base_choices = [
        correct_answer.to_string(),
        incorrect_answers[0].to_string(),
        incorrect_answers[1].to_string(),
        incorrect_answers[2].to_string(),
    ];

    let choices = (0..base_choices.len())
        .map(|idx| base_choices[(idx + rotation) % base_choices.len()].clone())
        .collect::<Vec<_>>();
    let answer_index = ((base_choices.len() - rotation) % base_choices.len()) as u8;

    (choices, answer_index)
}

pub fn join_subject_parts(parts: &[&str]) -> String {
    let joined = parts
        .iter()
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" / ");

    if joined.is_empty() {
        "general knowledge".to_string()
    } else {
        joined
    }
}
