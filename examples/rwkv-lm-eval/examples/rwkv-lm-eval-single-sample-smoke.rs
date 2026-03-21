use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use rwkv_config::get_arg_value;
use rwkv_config::validated::eval::EVAL_CFG;
use rwkv_eval::datasets::maths::set_llm_judger_semaphore;
use rwkv_eval::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, Field, get_benchmarks_with_field,
};
use rwkv_eval::init::init_cfg;
use rwkv_lm_eval::config_path::resolve_eval_cfg_path;
use rwkv_lm_eval::paths;
use tokio::sync::Semaphore;

fn load_eval_dotenv(config_dir: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(config_dir.join(".env"));
    if let Some(parent) = config_dir.parent() {
        candidates.push(parent.join(".env"));
    }
    candidates.push(paths::crate_root().join(".env"));
    candidates.push(PathBuf::from(".env"));

    candidates.into_iter().find_map(|path| {
        if !path.is_file() {
            return None;
        }
        dotenvy::from_path(&path)
            .unwrap_or_else(|err| panic!("failed to load .env from {}: {err}", path.display()));
        Some(path)
    })
}

fn norm_api_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    assert!(!base_url.is_empty(), "base_url cannot be empty");

    let base_url = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("http://{base_url}")
    };
    let base_url = base_url.trim_end_matches('/').to_string();

    if base_url.ends_with("/v1") {
        base_url
    } else {
        format!("{base_url}/v1")
    }
}

fn build_client(base_url: &str, api_key: &str) -> Client<OpenAIConfig> {
    let config = OpenAIConfig::new()
        .with_api_key(api_key.to_string())
        .with_api_base(norm_api_url(base_url));
    Client::with_config(config)
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

fn collect_benchmarks_from_config() -> Vec<&'static BenchmarkInfo> {
    let eval_cfg = EVAL_CFG.get().unwrap();
    let mut benchmark_infos = Vec::new();
    let mut seen = BTreeSet::new();

    for benchmark_field in &eval_cfg.benchmark_field {
        for benchmark_info in get_benchmarks_with_field(parse_field(benchmark_field)) {
            if seen.insert(benchmark_info.name.0) {
                benchmark_infos.push(*benchmark_info);
            }
        }
    }

    for benchmark_name in &eval_cfg.extra_benchmark_name {
        let benchmark_info = ALL_BENCHMARKS
            .iter()
            .find(|benchmark_info| benchmark_info.name.0 == benchmark_name)
            .unwrap_or_else(|| panic!("unknown benchmark `{benchmark_name}`"));
        if seen.insert(benchmark_info.name.0) {
            benchmark_infos.push(benchmark_info);
        }
    }

    benchmark_infos
}

async fn prepare_benchmark(benchmark_info: &BenchmarkInfo, benchmark: &mut dyn Benchmark) {
    let load_invalid = benchmark.load();
    let check_invalid = if load_invalid {
        true
    } else {
        benchmark.check().await
    };

    if load_invalid || check_invalid {
        println!("download benchmark: {}", benchmark_info.name.0);
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
            benchmark_info.name.0
        );
    }
}

fn preview(text: &str, limit: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(limit) {
        out.push(ch);
    }
    if text.chars().count() > limit {
        out.push_str("...");
    }
    out
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(paths::config_dir);
    let eval_cfg_name = get_arg_value(&args, "--eval-config")
        .unwrap_or_else(|| "polymath_arena_hard_v2_wmt24pp_smoke".into());
    let sample_index = get_arg_value(&args, "--sample-index")
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|err| panic!("invalid --sample-index `{value}`: {err}"))
        })
        .unwrap_or(0);
    let config_path = resolve_eval_cfg_path(&config_dir, &eval_cfg_name);
    let dotenv_path = load_eval_dotenv(&config_dir);

    let eval_cfg_builder = init_cfg(&config_dir, &eval_cfg_name);
    eval_cfg_builder.build();
    let eval_cfg = EVAL_CFG.get().unwrap();

    println!(
        "eval cfg: {eval_cfg_name} (config_dir: {})",
        config_dir.display()
    );
    println!(
        ".env: {}",
        dotenv_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<not found>".to_string())
    );
    println!("config path: {}", config_path.display());
    println!("datasets dir: {}", paths::datasets_path().display());
    println!("sample index: {}", sample_index);

    assert!(
        !eval_cfg.models.is_empty(),
        "smoke config requires at least one target model"
    );
    let target_model = &eval_cfg.models[0];
    let target_client = build_client(&target_model.base_url, &target_model.api_key);
    let judger_client = build_client(&eval_cfg.llm_judger.base_url, &eval_cfg.llm_judger.api_key);
    set_llm_judger_semaphore(Arc::new(Semaphore::new(1)));

    println!(
        "target model: {} @ {}",
        target_model.model, target_model.base_url
    );
    println!(
        "llm judger: {} @ {}",
        eval_cfg.llm_judger.model, eval_cfg.llm_judger.base_url
    );
    println!(
        "llm checker: {} @ {}",
        eval_cfg.llm_checker.model, eval_cfg.llm_checker.base_url
    );

    let benchmark_infos = collect_benchmarks_from_config();
    assert!(!benchmark_infos.is_empty(), "no benchmark selected");

    for benchmark_info in benchmark_infos {
        println!("\n=== benchmark: {} ===", benchmark_info.name.0);
        let mut benchmark = (benchmark_info.create)(paths::datasets_path());
        prepare_benchmark(benchmark_info, benchmark.as_mut()).await;
        assert!(
            benchmark.len() > sample_index,
            "benchmark `{}` has len={} < sample_index={sample_index}",
            benchmark_info.name.0,
            benchmark.len()
        );

        let cot_mode = *benchmark_info
            .cot_mode
            .first()
            .unwrap_or_else(|| panic!("benchmark `{}` has empty cot_mode", benchmark_info.name.0));
        let n_shot = *benchmark_info
            .n_shots
            .first()
            .unwrap_or_else(|| panic!("benchmark `{}` has empty n_shots", benchmark_info.name.0));

        let expected_context = benchmark.get_expected_context(sample_index, cot_mode, n_shot);
        let record = benchmark
            .answer_and_judge(
                &target_model.model,
                &target_client,
                benchmark_info
                    .with_llm_judger
                    .then_some(eval_cfg.llm_judger.model.as_str()),
                benchmark_info.with_llm_judger.then_some(&judger_client),
                cot_mode,
                n_shot,
                sample_index,
            )
            .await;

        println!("field: {:?}", benchmark_info.field);
        println!("display: {}", benchmark_info.display_name);
        println!("cot_mode: {:?}, n_shot: {}", cot_mode, n_shot);
        println!("prompt_preview:\n{}\n", preview(&expected_context, 1200));
        println!("answer_preview:\n{}\n", preview(&record.answer, 1200));
        println!(
            "ref_answer_preview:\n{}\n",
            preview(&record.ref_answer, 1200)
        );
        println!("is_passed: {}", record.is_passed);
        if !record.fail_reason.is_empty() {
            println!("fail_reason: {}", record.fail_reason);
        }
    }
}
