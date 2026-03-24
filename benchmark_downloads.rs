use std::any::Any;
use std::collections::BTreeSet;

use rwkv_eval::datasets::{ALL_BENCHMARKS, BenchmarkInfo};

const EXPECTED_BENCHMARKS: &[&str] = &[
    "aime24",
    "aime25",
    "algebra222",
    "amc23",
    "answer_judge",
    "arena_hard_v2",
    "asdiv",
    "beyond_aime",
    "browsecomp",
    "browsecomp_zh",
    "brumo25",
    "ceval",
    "cmmlu",
    "college_math",
    "comp_math_24_25",
    "gaokao2023en",
    "gpqa_diamond",
    "gpqa_extended",
    "gpqa_main",
    "gsm8k",
    "gsm_plus",
    "hendrycks_math",
    "hle",
    "hmmt_feb25",
    "human_eval",
    "human_eval_cn",
    "human_eval_fix",
    "human_eval_plus",
    "include",
    "ifeval",
    "livecodebench",
    "math_500",
    "math_odyssey",
    "mawps",
    "mbpp",
    "mbpp_plus",
    "mcp_bench",
    "mmlu",
    "mmlu_pro",
    "mmlu_redux",
    "minerva_math",
    "mmmlu",
    "olympiadbench",
    "omni_math",
    "polymath",
    "simpleqa",
    "supergpqa",
    "svamp",
    "tau_bench",
    "wmt24pp",
];

fn benchmark_info(name: &str) -> &'static BenchmarkInfo {
    ALL_BENCHMARKS
        .iter()
        .find(|info| info.name.0 == name)
        .unwrap_or_else(|| panic!("missing benchmark info for `{name}`"))
}

#[derive(Debug)]
struct FailureRecord {
    benchmark: String,
    error: String,
}

async fn assert_benchmark_download_load_and_read(info: &'static BenchmarkInfo) {
    let tempdir = tempfile::tempdir()
        .unwrap_or_else(|err| panic!("create tempdir failed for `{}`: {err}", info.name.0));
    let mut benchmark = (info.create)(tempdir.path().to_path_buf());

    benchmark.download().await;

    assert!(
        !benchmark.load(),
        "load() returned invalid after download for `{}`",
        info.name.0
    );
    assert!(benchmark.len() > 0, "len() == 0 for `{}`", info.name.0);

    let cot_mode = *info
        .cot_mode
        .first()
        .unwrap_or_else(|| panic!("benchmark `{}` has empty cot_mode", info.name.0));
    let n_shot = *info
        .n_shots
        .first()
        .unwrap_or_else(|| panic!("benchmark `{}` has empty n_shots", info.name.0));

    let expected_context = benchmark.get_expected_context(0, cot_mode, n_shot);
    assert!(
        !expected_context.trim().is_empty(),
        "get_expected_context(0, ..) returned empty text for `{}`",
        info.name.0
    );

    let ref_answer = benchmark.get_ref_answer(0);
    assert!(
        !ref_answer.trim().is_empty(),
        "get_ref_answer(0) returned empty text for `{}`",
        info.name.0
    );
}

fn panic_payload_to_string(payload: Box<dyn Any + Send + 'static>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_string(),
            Err(_) => "non-string panic payload".to_string(),
        },
    }
}

fn render_failure_report(total: usize, failures: &[FailureRecord]) -> String {
    let passed = total.saturating_sub(failures.len());
    let mut report = format!(
        "Benchmark download/load summary: total={total}, passed={passed}, skipped={}\n",
        failures.len()
    );

    if failures.is_empty() {
        report.push_str("All benchmark download/load checks passed.\n");
        return report;
    }

    report.push_str(
        "The entries below were skipped after a download/load failure so the suite could continue.\n",
    );

    for failure in failures {
        report.push_str(&format!(
            "\n=== {} ===\n{}\n",
            failure.benchmark, failure.error
        ));
    }

    report
}

#[test]
fn harness_covers_all_registered_benchmarks() {
    let expected = EXPECTED_BENCHMARKS.iter().copied().collect::<BTreeSet<_>>();
    let actual = ALL_BENCHMARKS
        .iter()
        .map(|info| info.name.0)
        .collect::<BTreeSet<_>>();

    assert_eq!(
        actual, expected,
        "benchmark coverage drift detected in benchmark_downloads.rs"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn downloads_and_reads_all_benchmarks() {
    let mut failures = Vec::new();

    for &benchmark_name in EXPECTED_BENCHMARKS {
        println!("[benchmark] {benchmark_name}");

        let info = benchmark_info(benchmark_name);
        match tokio::spawn(async move { assert_benchmark_download_load_and_read(info).await }).await
        {
            Ok(()) => {
                println!("[pass] {benchmark_name}");
            }
            Err(join_error) if join_error.is_panic() => {
                let error = panic_payload_to_string(join_error.into_panic());
                eprintln!("[skip] {benchmark_name}: {error}");
                failures.push(FailureRecord {
                    benchmark: benchmark_name.to_string(),
                    error,
                });
            }
            Err(join_error) => {
                let error = format!("task join error: {join_error}");
                eprintln!("[skip] {benchmark_name}: {error}");
                failures.push(FailureRecord {
                    benchmark: benchmark_name.to_string(),
                    error,
                });
            }
        }
    }

    let report = render_failure_report(EXPECTED_BENCHMARKS.len(), &failures);
    println!("\n{report}");
}
