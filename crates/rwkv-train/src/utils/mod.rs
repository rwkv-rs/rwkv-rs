use std::{
    fs::{create_dir_all, read_dir},
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::AutodiffBackend,
};
use dialoguer::Confirm;
use regex::Regex;

pub fn auto_create_directory(path: PathBuf) -> PathBuf {
    if !path.exists() {
        create_dir_all(&path).unwrap();
    }

    path
}

pub fn read_record_file(
    record_path: Option<String>,
    full_experiment_log_path: &Path,
) -> Option<String> {
    let full_records_dir_path = auto_create_directory(full_experiment_log_path.join("records"));

    match &record_path {
        Some(record_path) => {
            let full_record_path = PathBuf::from(record_path).canonicalize().unwrap();

            if full_record_path.starts_with(&full_records_dir_path)
                && let Some(full_last_record_path) = find_last_record_path(&full_records_dir_path)
                && full_record_path != full_last_record_path
            {
                let prompt = format!(
                    "⚠️ 你指定了 {}，但 record 目录下最新的是 {}。\n若想自动加载最新，请留空 \
                     record_path；\n确认继续加载指定文件？",
                    full_record_path.display(),
                    full_last_record_path.display(),
                );

                if !Confirm::new()
                    .with_prompt(prompt)
                    .default(false)
                    .interact()
                    .expect("读取用户输入失败")
                {
                    panic!("用户取消，程序终止");
                }
            }

            Some(full_record_path.to_string_lossy().to_string())
        }
        None => find_last_record_path(&full_records_dir_path)
            .map(|last_record_path| last_record_path.to_string_lossy().to_string()),
    }
}

fn find_last_record_path(full_records_dir_path: &Path) -> Option<PathBuf> {
    let re = Regex::new(r"^E(\d+)S(\d+)\.mpk$").unwrap();

    read_dir(full_records_dir_path)
        .unwrap()
        .flatten()
        .filter_map(|entry| {
            let os_name = entry.file_name();

            let name = os_name.to_string_lossy();

            re.captures(&name).map(|cap| {
                let miniepoch: usize = cap[1].parse::<usize>().unwrap();

                let step: usize = cap[2].parse::<usize>().unwrap();

                ((miniepoch, step), entry.path().canonicalize().unwrap())
            })
        })
        .max_by_key(|((miniepoch, step), _)| (*miniepoch, *step))
        .map(|(_, path)| {
            println!("🔄 自动加载最新权重: {}", path.display());

            path
        })
}

pub fn save_model_weights<B: AutodiffBackend, M: Module<B>>(
    model: &M,
    full_experiment_log_path: &Path,
    mini_epoch: usize,
    step: usize,
) {
    let full_records_dir_path = auto_create_directory(full_experiment_log_path.join("records"));

    let filename = format!("E{}S{}.mpk", mini_epoch, step);

    let save_path = full_records_dir_path.join(filename);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    model
        .clone()
        .save_file(save_path.clone(), &recorder)
        .unwrap();
}
