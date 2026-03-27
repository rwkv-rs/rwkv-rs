use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rwkv_config::validated::eval::EVAL_CFG;
use rwkv_eval::datasets::{ALL_BENCHMARKS, Benchmark, BenchmarkInfo, Field, get_benchmarks_with_field};
use rwkv_eval::evaluators::coding::{
    ensure_microsandbox_available, ensure_microsandbox_runtime_dependencies,
};
use tokio::time::{Duration, sleep};

const MSB_PATH_ENV: &str = "MSB_PATH";
const RWKV_EVAL_MSB_PATH_ENV: &str = "RWKV_EVAL_MSB_PATH";
const RWKV_EVAL_SKIP_DATASET_CHECK_ENV: &str = "RWKV_EVAL_SKIP_DATASET_CHECK";

pub(crate) fn collect_benchmarks() -> Vec<&'static BenchmarkInfo> {
    let mut benchmark_infos = BTreeMap::new();

    for benchmark_field in &EVAL_CFG.get().unwrap().benchmark_field {
        for benchmark_info in get_benchmarks_with_field(parse_field(benchmark_field)) {
            benchmark_infos.insert(benchmark_info.name.0, *benchmark_info);
        }
    }

    for benchmark_name in &EVAL_CFG.get().unwrap().extra_benchmark_name {
        let benchmark_info = ALL_BENCHMARKS
            .iter()
            .find(|benchmark_info| benchmark_info.name.0 == benchmark_name)
            .unwrap_or_else(|| panic!("unknown benchmark `{benchmark_name}`"));
        benchmark_infos.insert(benchmark_info.name.0, benchmark_info);
    }

    benchmark_infos.into_values().collect()
}

pub(crate) fn validate_benchmark_info(benchmark_info: &BenchmarkInfo) {
    assert!(
        !benchmark_info.avg_ks.is_empty(),
        "benchmark `{}` has empty avg_ks",
        benchmark_info.name.0,
    );
    assert!(
        !benchmark_info.pass_ks.is_empty(),
        "benchmark `{}` has empty pass_ks",
        benchmark_info.name.0,
    );
}

pub(crate) async fn prepare_benchmark(
    benchmark_info: &BenchmarkInfo,
    benchmark: &mut dyn Benchmark,
) {
    let skip_dataset_check = skip_dataset_check();
    if skip_dataset_check {
        println!(
            "dataset check: skipped for {} via {}",
            benchmark_info.name.0, RWKV_EVAL_SKIP_DATASET_CHECK_ENV
        );
    }
    let load_invalid = benchmark.load();
    let check_invalid = if load_invalid {
        true
    } else if skip_dataset_check {
        false
    } else {
        benchmark.check().await
    };

    if load_invalid || check_invalid {
        benchmark.download().await;

        let load_invalid = benchmark.load();
        let check_invalid = if load_invalid {
            true
        } else if skip_dataset_check {
            false
        } else {
            benchmark.check().await
        };

        assert!(
            !load_invalid && !check_invalid,
            "benchmark `{}` is still invalid after download",
            benchmark_info.name.0,
        );
    }
}

fn skip_dataset_check() -> bool {
    std::env::var(RWKV_EVAL_SKIP_DATASET_CHECK_ENV)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(crate) async fn ensure_microsandbox_for_coding_benchmarks(
    benchmark_infos: &[&'static BenchmarkInfo],
) {
    if !benchmark_infos
        .iter()
        .any(|benchmark_info| benchmark_info.field == Field::Coding)
    {
        return;
    }

    configure_msb_path_env();
    ensure_microsandbox_runtime_dependencies()
        .await
        .unwrap_or_else(|err| {
            panic!(
                "coding benchmark requires microsandbox runtime dependencies, but automatic installation failed: {err}. \
please verify the host can download the official libkrunfw release for microsandbox 0.3.0."
            )
        });

    let initial_err = match ensure_microsandbox_available().await {
        Ok(()) => return,
        Err(err) => err,
    };

    start_microsandbox_service().unwrap_or_else(|err| {
        panic!(
            "coding benchmark requires microsandbox, but automatic startup failed. \
initial probe error: {initial_err}. startup error: {err}. \
please install the real microsandbox CLI/runtime, then set `MSB_PATH` or `RWKV_EVAL_MSB_PATH` to the `msb` binary if it is not already on PATH."
        )
    });

    let mut last_err = String::new();
    for _ in 0..20 {
        match ensure_microsandbox_available().await {
            Ok(()) => return,
            Err(err) => last_err = err,
        }
        sleep(Duration::from_millis(500)).await;
    }

    panic!(
        "coding benchmark requires microsandbox, but the service is still unavailable after automatic startup: {}. \
please check microsandbox installation and server startup logs.",
        last_err
    );
}

fn start_microsandbox_service() -> Result<(), String> {
    let mut errors = Vec::new();

    if let Some(msb_path) = discover_msb_cli_path() {
        match spawn_detached(&msb_path, &["server", "start", "--dev", "--detach"]) {
            Ok(()) => return Ok(()),
            Err(err) => {
                errors.push(format!(
                    "`{} server start --dev --detach`: {err}",
                    msb_path.display()
                ));
            }
        }
    } else {
        errors.push(format!(
            "no `msb` binary found (searched {})",
            describe_msb_search_locations()
        ));
    }

    for server_path in discover_microsandbox_server_paths() {
        match spawn_detached(&server_path, &[]) {
            Ok(()) => return Ok(()),
            Err(err) => errors.push(format!("`{}`: {err}", server_path.display())),
        }
    }

    Err(errors.join("; "))
}

fn configure_msb_path_env() {
    if std::env::var_os(MSB_PATH_ENV).is_some() {
        return;
    }

    let Some(msb_path) = discover_microsandbox_runtime_path() else {
        return;
    };

    unsafe {
        std::env::set_var(MSB_PATH_ENV, &msb_path);
    }
    println!("microsandbox: using runtime at {}", msb_path.display());
}

fn discover_microsandbox_runtime_path() -> Option<PathBuf> {
    configured_binary_paths()
        .into_iter()
        .chain(discover_binary_paths("msb"))
        .chain(discover_binary_paths("msbrun"))
        .find(|path| path_supports_subcommand(path, "supervisor"))
}

fn discover_msb_cli_path() -> Option<PathBuf> {
    configured_binary_paths()
        .into_iter()
        .chain(discover_binary_paths("msb"))
        .find(|path| path_supports_subcommand(path, "server"))
}

fn configured_binary_paths() -> Vec<PathBuf> {
    [
        std::env::var_os(RWKV_EVAL_MSB_PATH_ENV).map(PathBuf::from),
        std::env::var_os(MSB_PATH_ENV).map(PathBuf::from),
    ]
    .into_iter()
    .flatten()
    .filter(|path| is_executable_file(path))
    .collect()
}

fn discover_binary_paths(program: &str) -> Vec<PathBuf> {
    [
        std::env::var_os("VIRTUAL_ENV")
            .map(PathBuf::from)
            .map(|root| root.join("bin").join(program)),
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(".venv").join("bin").join(program)),
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join("target").join("microsandbox").join(program)),
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local").join("bin").join(program)),
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".microsandbox").join("bin").join(program)),
        find_on_path(program),
    ]
    .into_iter()
    .flatten()
    .filter(|path| is_executable_file(path))
    .collect()
}

fn discover_microsandbox_server_paths() -> Vec<PathBuf> {
    [
        std::env::var_os("VIRTUAL_ENV")
            .map(PathBuf::from)
            .map(|root| root.join("bin").join("msbserver")),
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(".venv").join("bin").join("msbserver")),
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local").join("bin").join("msbserver")),
        find_on_path("msbserver"),
        std::env::var_os("VIRTUAL_ENV")
            .map(PathBuf::from)
            .map(|root| root.join("bin").join("microsandbox-server")),
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(".venv").join("bin").join("microsandbox-server")),
        find_on_path("microsandbox-server"),
    ]
    .into_iter()
    .flatten()
    .filter(|path| is_executable_file(path))
    .collect()
}

fn spawn_detached(program: &Path, args: &[&str]) -> Result<(), std::io::Error> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command.spawn().map(|_| ())
}

fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

fn path_supports_subcommand(path: &Path, subcommand: &str) -> bool {
    let Ok(status) = Command::new(path)
        .arg(subcommand)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    else {
        return false;
    };

    status.success()
}

fn find_on_path(program: &str) -> Option<PathBuf> {
    let path_value = std::env::var_os("PATH")?;
    std::env::split_paths(&path_value)
        .map(|dir| dir.join(program))
        .find(|candidate| is_executable_file(candidate))
}

fn describe_msb_search_locations() -> String {
    [
        format!("{RWKV_EVAL_MSB_PATH_ENV}=<path>"),
        format!("{MSB_PATH_ENV}=<path>"),
        "$VIRTUAL_ENV/bin/msb".to_string(),
        "$VIRTUAL_ENV/bin/msbrun".to_string(),
        "./.venv/bin/msb".to_string(),
        "./.venv/bin/msbrun".to_string(),
        "./target/microsandbox/msb".to_string(),
        "./target/microsandbox/msbrun".to_string(),
        "~/.local/bin/msb".to_string(),
        "~/.local/bin/msbrun".to_string(),
        "~/.microsandbox/bin/msb".to_string(),
        "~/.microsandbox/bin/msbrun".to_string(),
        "PATH".to_string(),
    ]
    .join(", ")
}

fn parse_field(field_name: &str) -> Field {
    match field_name.trim() {
        "Knowledge" => Field::Knowledge,
        "Math" | "Maths" => Field::Maths,
        "Coding" => Field::Coding,
        "Instruction Following" | "InstructionFollowing" => Field::InstructionFollowing,
        "Function Call" | "FunctionCalling" => Field::FunctionCalling,
        _ => panic!("unknown benchmark field `{field_name}`"),
    }
}
