use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rwkv::custom::backend::Cuda;
use rwkv::custom::cubecl::device::{Device as CubeDevice, DeviceId};
use rwkv::custom::prelude::Backend;
use rwkv::custom::tensor::bf16;
use rwkv_lm::model::AutoRegressiveModelConfig;
use rwkv_lm::pth2mpk::{ConvertPthToMpkOptions, convert_pth_to_mpk};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

const VOCAB_SIZE: usize = 65536;
const HEAD_SIZE: usize = 64;
const TASKS_PER_DEVICE: usize = 2;

struct ConvertJob {
    model_path: PathBuf,
    output_path: PathBuf,
    model_config: AutoRegressiveModelConfig,
}

#[derive(Clone)]
struct ParsedCheckpoint {
    model_path: PathBuf,
    file_stem: String,
    data_version: String,
    model_size: String,
    date: String,
}

struct JobBuildResult {
    total_pth_files: usize,
    skipped_existing_outputs: usize,
    skipped_arch_mismatch: usize,
    skipped_older_same_size: usize,
    jobs: Vec<ConvertJob>,
}

struct WorkerPlan {
    device_id: DeviceId,
    jobs: Vec<ConvertJob>,
}

#[derive(Default)]
struct WorkerStats {
    converted: usize,
    skipped_failed_conversion: usize,
}

fn infer_model_config(model_size: &str) -> io::Result<AutoRegressiveModelConfig> {
    let (num_cells, embedded_dim) = if model_size == "0.1b" {
        (12, 768)
    } else if model_size == "0.4b" {
        (24, 1024)
    } else if model_size == "1.5b" {
        (24, 2048)
    } else if model_size == "2.9b" {
        (32, 2560)
    } else if model_size == "7.2b" {
        (32, 4096)
    } else if model_size == "13.3b" {
        (61, 4096)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported model size in file name: {model_size}"),
        ));
    };

    let num_heads = embedded_dim / HEAD_SIZE;
    Ok(AutoRegressiveModelConfig::new(
        num_cells,
        VOCAB_SIZE,
        embedded_dim,
        num_heads,
        HEAD_SIZE,
    ))
}

fn parse_rwkv7_checkpoint(
    file_stem: &str,
    model_path: PathBuf,
) -> io::Result<Option<ParsedCheckpoint>> {
    let arch_version = file_stem
        .split('-')
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing file stem"))?;

    if arch_version != "rwkv7" {
        return Ok(None);
    }

    let mut parts = file_stem.split('-');
    let _arch_version = parts.next();
    let data_version = parts.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid rwkv7 checkpoint name: {file_stem}"),
        )
    })?;
    let model_size = parts.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid rwkv7 checkpoint name: {file_stem}"),
        )
    })?;
    let date = parts.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid rwkv7 checkpoint name: {file_stem}"),
        )
    })?;

    if date.len() != 8 || !date.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid rwkv7 checkpoint date: {file_stem}"),
        ));
    }

    Ok(Some(ParsedCheckpoint {
        model_path,
        file_stem: file_stem.to_string(),
        data_version: data_version.to_string(),
        model_size: model_size.to_string(),
        date: date.to_string(),
    }))
}

fn collect_pth_files(input_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut model_paths = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("pth") {
            model_paths.push(path);
        }
    }

    model_paths.sort();

    if model_paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no .pth files found in {}", input_dir.display()),
        ));
    }

    Ok(model_paths)
}

fn is_newer_checkpoint(candidate: &ParsedCheckpoint, current: &ParsedCheckpoint) -> bool {
    (
        candidate.date.as_str(),
        candidate.data_version.as_str(),
        candidate.file_stem.as_str(),
    ) > (
        current.date.as_str(),
        current.data_version.as_str(),
        current.file_stem.as_str(),
    )
}

fn build_jobs(model_paths: Vec<PathBuf>, output_dir: &Path) -> io::Result<JobBuildResult> {
    let total_pth_files = model_paths.len();
    let mut skipped_existing_outputs = 0;
    let mut skipped_arch_mismatch = 0;
    let mut latest_by_model_size: HashMap<String, ParsedCheckpoint> = HashMap::new();
    let mut jobs = Vec::new();

    for model_path in model_paths {
        let file_stem = model_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid utf-8 file stem: {}", model_path.display()),
                )
            })?
            .to_string();

        let Some(parsed) = parse_rwkv7_checkpoint(&file_stem, model_path)? else {
            skipped_arch_mismatch += 1;
            println!("skipping {}: arch_version is not rwkv7", file_stem);
            continue;
        };

        match latest_by_model_size.get_mut(&parsed.model_size) {
            Some(current) => {
                if is_newer_checkpoint(&parsed, current) {
                    *current = parsed;
                }
            }
            None => {
                latest_by_model_size.insert(parsed.model_size.clone(), parsed);
            }
        }
    }

    let skipped_older_same_size = total_pth_files
        .saturating_sub(skipped_arch_mismatch)
        .saturating_sub(latest_by_model_size.len());

    let mut selected = latest_by_model_size.into_values().collect::<Vec<_>>();
    selected.sort_by(|left, right| {
        left.model_size
            .cmp(&right.model_size)
            .then(left.date.cmp(&right.date))
            .then(left.data_version.cmp(&right.data_version))
            .then(left.file_stem.cmp(&right.file_stem))
    });

    for parsed in selected {
        let output_path = output_dir.join(format!("{}.mpk", parsed.file_stem));
        if output_path.exists() {
            skipped_existing_outputs += 1;
            println!(
                "skipping {}: output already exists at {}",
                parsed.model_path.display(),
                output_path.display()
            );
            continue;
        }

        let model_config = infer_model_config(&parsed.model_size)?;
        jobs.push(ConvertJob {
            model_path: parsed.model_path,
            output_path,
            model_config,
        });
    }

    Ok(JobBuildResult {
        total_pth_files,
        skipped_existing_outputs,
        skipped_arch_mismatch,
        skipped_older_same_size,
        jobs,
    })
}

fn build_worker_plans(jobs: Vec<ConvertJob>, slot_devices: &[DeviceId]) -> Vec<WorkerPlan> {
    let mut workers = slot_devices
        .iter()
        .copied()
        .map(|device_id| WorkerPlan {
            device_id,
            jobs: Vec::new(),
        })
        .collect::<Vec<_>>();

    let worker_count = workers.len();
    for (index, job) in jobs.into_iter().enumerate() {
        workers[index % worker_count].jobs.push(job);
    }

    workers.retain(|worker| !worker.jobs.is_empty());
    workers
}

fn device_slots(device_count: usize) -> Vec<DeviceId> {
    let mut slots = Vec::with_capacity(device_count * TASKS_PER_DEVICE);
    for device_index in 0..device_count {
        for _ in 0..TASKS_PER_DEVICE {
            slots.push(DeviceId::new(0, device_index as u32));
        }
    }
    slots
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn main() -> io::Result<()> {
    type MyBackend = Cuda<bf16, i32>;
    let args = env::args().collect::<Vec<_>>();
    let input_dir = args
        .windows(2)
        .find(|window| window[0] == "--input-dir")
        .map(|window| PathBuf::from(&window[1]))
        .or_else(|| env::var_os("RWKV_PTH_INPUT_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/public/home/ssjxzkz/Weights/BlinkDL__rwkv7-g1"));
    let output_dir = args
        .windows(2)
        .find(|window| window[0] == "--output-dir")
        .map(|window| PathBuf::from(&window[1]))
        .or_else(|| env::var_os("RWKV_MPK_OUTPUT_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/public/home/ssjxzkz/Weights/Caizus__rwkv-rs-g1"));

    println!("input dir: {}", input_dir.display());
    println!("output dir: {}", output_dir.display());

    fs::create_dir_all(&output_dir)?;

    let device_count = <<MyBackend as Backend>::Device as CubeDevice>::device_count_total();
    if device_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no CUDA devices found",
        ));
    }

    let slot_devices = device_slots(device_count);
    println!("detected CUDA devices: {device_count}");
    println!("parallel slots: {}", slot_devices.len());

    let result = build_jobs(collect_pth_files(&input_dir)?, &output_dir)?;
    let job_count = result.jobs.len();

    println!("found {} .pth file(s)", result.total_pth_files);
    println!(
        "skipped {} file(s) due to arch mismatch",
        result.skipped_arch_mismatch
    );
    println!(
        "skipped {} file(s) because a newer checkpoint exists for the same model size",
        result.skipped_older_same_size
    );
    println!(
        "skipped {} file(s) due to existing output",
        result.skipped_existing_outputs
    );

    if job_count == 0 {
        println!("no conversion jobs to run");
        return Ok(());
    }

    let workers = build_worker_plans(result.jobs, &slot_devices);
    let worker_count = workers.len();
    println!("scheduled {job_count} conversion job(s) across {worker_count} worker slot(s)");

    let pool = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|error| io::Error::other(format!("failed to build rayon thread pool: {error}")))?;

    let stats = pool.install(|| {
        workers
            .into_par_iter()
            .map(|worker| {
                let mut stats = WorkerStats::default();
                for job in worker.jobs {
                    println!(
                        "converting {} -> {} on cuda:{}",
                        job.model_path.display(),
                        job.output_path.display(),
                        worker.device_id.index_id
                    );

                    let option = ConvertPthToMpkOptions::new(
                        job.model_path.to_string_lossy().into_owned(),
                        job.output_path.to_string_lossy().into_owned(),
                    )
                    .with_device_id(worker.device_id);
                    let result = panic::catch_unwind(AssertUnwindSafe(|| {
                        convert_pth_to_mpk::<MyBackend>(&option, job.model_config)
                    }));

                    match result {
                        Ok(Ok(())) => {
                            stats.converted += 1;
                        }
                        Ok(Err(error)) => {
                            stats.skipped_failed_conversion += 1;
                            if job.output_path.exists() {
                                let _ = fs::remove_file(&job.output_path);
                            }
                            eprintln!(
                                "skipping {} after conversion failure on cuda:{}: {}",
                                job.model_path.display(),
                                worker.device_id.index_id,
                                error
                            );
                        }
                        Err(payload) => {
                            stats.skipped_failed_conversion += 1;
                            if job.output_path.exists() {
                                let _ = fs::remove_file(&job.output_path);
                            }
                            eprintln!(
                                "skipping {} after panic on cuda:{}: {}",
                                job.model_path.display(),
                                worker.device_id.index_id,
                                panic_payload_to_string(payload)
                            );
                        }
                    }
                }
                stats
            })
            .reduce(WorkerStats::default, |mut left, right| {
                left.converted += right.converted;
                left.skipped_failed_conversion += right.skipped_failed_conversion;
                left
            })
    });

    println!("converted {} file(s)", stats.converted);
    println!(
        "skipped {} file(s) due to conversion failure",
        stats.skipped_failed_conversion
    );
    Ok(())
}
