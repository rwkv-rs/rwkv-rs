use std::{
    collections::BTreeMap,
    process::{Command, Stdio},
};

use rwkv_config::validated::eval::EVAL_CFG;
use tokio::time::{Duration, sleep};

use crate::cores::{
    datasets::{ALL_BENCHMARKS, Benchmark, BenchmarkInfo, Field, get_benchmarks_with_field},
    evaluators::coding::ensure_microsandbox_available,
};

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
    let load_invalid = benchmark.load();
    let check_invalid = if load_invalid {
        true
    } else {
        benchmark.check().await
    };

    if load_invalid || check_invalid {
        benchmark.download().await;

        let load_invalid = benchmark.load();
        let check_invalid = if load_invalid {
            true
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

pub(crate) async fn ensure_microsandbox_for_coding_benchmarks(
    benchmark_infos: &[&'static BenchmarkInfo],
) {
    if !benchmark_infos
        .iter()
        .any(|benchmark_info| benchmark_info.field == Field::Coding)
    {
        return;
    }

    if ensure_microsandbox_available().await.is_ok() {
        return;
    }

    start_microsandbox_service().unwrap_or_else(|err| {
        panic!(
            "coding benchmark requires microsandbox, but automatic startup failed: {err}. \
please check microsandbox installation and ensure `msb server start --dev` or `microsandbox-server` is available."
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

    for (program, args) in [
        ("msb", &["server", "start", "--dev"][..]),
        ("microsandbox-server", &[][..]),
    ] {
        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        match command.spawn() {
            Ok(_) => return Ok(()),
            Err(err) => {
                let rendered_args = if args.is_empty() {
                    String::new()
                } else {
                    format!(" {}", args.join(" "))
                };
                errors.push(format!("`{program}{rendered_args}`: {err}"));
            }
        }
    }

    Err(errors.join("; "))
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
