use async_openai::Client;
use async_openai::config::OpenAIConfig;
use rwkv_config::raw::eval::ApiConfig;
use rwkv_config::validated::eval::{EVAL_CFG, FinalEvalConfigBuilder};
use rwkv_eval::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, Field, SamplingConfig, get_benchmarks_with_field,
};
use rwkv_eval::evaluators::coding::ensure_microsandbox_available;
use rwkv_eval::inferers::{CompletionRequest, CompletionResponse};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tokio::time::{Duration, sleep};

const AVG_K_SAMPLE_BASE_SEED: u64 = 0xA11CE5EED5EED123;

struct ClientWithConfig {
    api_cfg: ApiConfig,
    client: Client<OpenAIConfig>,
}

struct AvgKExecutionPlan {
    repeat_count: usize,
    indices: Vec<usize>,
}

pub async fn evaluating(eval_cfg_builder: FinalEvalConfigBuilder, datasets_path: PathBuf) {
    eval_cfg_builder.build();

    let experiment_name = EVAL_CFG.get().unwrap().experiment_name.clone();
    let target_models = collect_models();
    assert!(
        !target_models.is_empty(),
        "no target model matched model_arch_versions/model_data_versions/model_num_params"
    );

    let clients_with_cfg = target_models
        .into_iter()
        .map(|api_cfg| ClientWithConfig {
            client: build_client(&api_cfg),
            api_cfg,
        })
        .collect::<Vec<_>>();
    let llm_judger_cfg = EVAL_CFG.get().unwrap().llm_judger.clone();
    let llm_checker_cfg = EVAL_CFG.get().unwrap().llm_checker.clone();
    let llm_judger_client = build_client(&llm_judger_cfg);
    let llm_checker_client = build_client(&llm_checker_cfg);

    println!("experiment: {experiment_name}");
    println!("target models: {}", clients_with_cfg.len());

    for target_model in &clients_with_cfg {
        check_client(&target_model.client, &target_model.api_cfg).await;
    }
    check_client(&llm_judger_client, &llm_judger_cfg).await;
    check_client(&llm_checker_client, &llm_checker_cfg).await;

    let benchmark_infos = collect_benchmarks();
    assert!(!benchmark_infos.is_empty(), "no benchmark selected");
    ensure_microsandbox_for_coding_benchmarks(&benchmark_infos).await;

    for benchmark_info in benchmark_infos {
        println!("prepare benchmark: {}", benchmark_info.name.0);
        let mut benchmark = (benchmark_info.create)(datasets_path.clone());
        prepare_benchmark(benchmark_info, benchmark.as_mut());

        let max_pass_k = benchmark_info
            .pass_ks
            .iter()
            .copied()
            .max()
            .unwrap_or_else(|| panic!("benchmark `{}` has empty pass_ks", benchmark_info.name.0));
        let judger_client = benchmark_info.with_llm_judger.then_some(&llm_judger_client);
        let judger_model_name = benchmark_info
            .with_llm_judger
            .then_some(llm_judger_cfg.model.as_str());

        for target_model in &clients_with_cfg {
            println!(
                "run benchmark={} model={}",
                benchmark_info.name.0, target_model.api_cfg.model,
            );

            for &cot_mode in benchmark_info.cot_mode {
                for &n_shot in benchmark_info.n_shots {
                    for &avg_k in benchmark_info.avg_ks {
                        let avg_k_plan = build_avg_k_execution_plan(
                            benchmark_info.name.0,
                            benchmark.len(),
                            avg_k,
                        );
                        let mut raw_success_counts = Vec::with_capacity(avg_k_plan.repeat_count);

                        for _ in 0..avg_k_plan.repeat_count {
                            let mut success_counts = Vec::with_capacity(avg_k_plan.indices.len());

                            for &index in &avg_k_plan.indices {
                                let mut success_count = 0u8;
                                for _ in 0..max_pass_k {
                                    if benchmark
                                        .answer_and_judge(
                                            &target_model.api_cfg.model,
                                            &target_model.client,
                                            judger_model_name,
                                            judger_client,
                                            cot_mode,
                                            n_shot,
                                            index,
                                        )
                                        .await
                                    {
                                        success_count += 1;
                                    }
                                }
                                success_counts.push(success_count);
                            }

                            raw_success_counts.push(success_counts);
                        }

                        let passed = raw_success_counts
                            .iter()
                            .flatten()
                            .filter(|&&success_count| success_count > 0)
                            .count();
                        let total = avg_k_plan.repeat_count * avg_k_plan.indices.len();

                        println!(
                            "  cot_mode={:?} n_shot={} avg_k={} sample_size={} repeats={} pass_k_max={} passed={}/{}",
                            cot_mode,
                            n_shot,
                            avg_k,
                            avg_k_plan.indices.len(),
                            avg_k_plan.repeat_count,
                            max_pass_k,
                            passed,
                            total,
                        );

                        // 预留逻辑: 后面会把这里的逐题原始结果和聚合结果保存到数据库。
                    }
                }
            }
        }
    }
}

fn collect_models() -> Vec<ApiConfig> {
    let mut target_models = Vec::new();
    for model_arch_version in &EVAL_CFG.get().unwrap().model_arch_versions {
        for model_data_version in &EVAL_CFG.get().unwrap().model_data_versions {
            for model_num_param in &EVAL_CFG.get().unwrap().model_num_params {
                target_models.extend(
                    EVAL_CFG
                        .get()
                        .unwrap()
                        .models
                        .iter()
                        .filter(|model| {
                            model.model_arch_version == *model_arch_version
                                && model.model_data_version == *model_data_version
                                && model.model_num_params == *model_num_param
                        })
                        .cloned(),
                );
            }
        }
    }

    target_models
}

fn collect_benchmarks() -> Vec<&'static BenchmarkInfo> {
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

async fn ensure_microsandbox_for_coding_benchmarks(benchmark_infos: &[&'static BenchmarkInfo]) {
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

fn build_client(api_cfg: &ApiConfig) -> Client<OpenAIConfig> {
    let config = OpenAIConfig::new()
        .with_api_key(api_cfg.api_key.clone())
        .with_api_base(norm_api_url(&api_cfg.base_url));

    Client::with_config(config)
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

async fn check_client(client: &Client<OpenAIConfig>, api_cfg: &ApiConfig) {
    let req = CompletionRequest::new(
        api_cfg.model.clone(),
        "ping".into(),
        vec!["\n".to_string()],
        1,
        &SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            presence_penalty: 0.0,
            repetition_penalty: 0.0,
            penalty_decay: 1.0,
        },
        None,
        None,
    );

    let _: CompletionResponse = client
        .completions()
        .create_byot(&req)
        .await
        .unwrap_or_else(|error| panic!("client `{}` is unavailable: {error}", api_cfg.model));
}

fn build_avg_k_execution_plan(
    benchmark_name: &str,
    benchmark_len: usize,
    avg_k: f32,
) -> AvgKExecutionPlan {
    assert!(
        avg_k.is_finite() && avg_k > 0.0,
        "benchmark `{benchmark_name}` has invalid avg_k={avg_k}; avg_k must be finite and > 0"
    );
    assert!(
        benchmark_len > 0,
        "benchmark `{benchmark_name}` has no samples to evaluate"
    );

    if avg_k < 1.0 {
        let sample_size = compute_ratio_sample_size(benchmark_len, avg_k);
        let seed = AVG_K_SAMPLE_BASE_SEED
            ^ fnv1a_hash64(benchmark_name.as_bytes())
            ^ u64::from(avg_k.to_bits());
        AvgKExecutionPlan {
            repeat_count: 1,
            indices: deterministic_sample_indices(benchmark_len, sample_size, seed),
        }
    } else {
        let repeat_count = parse_avg_k_repeat_count(benchmark_name, avg_k);
        AvgKExecutionPlan {
            repeat_count,
            indices: (0..benchmark_len).collect(),
        }
    }
}

fn parse_avg_k_repeat_count(benchmark_name: &str, avg_k: f32) -> usize {
    let rounded = avg_k.round();
    assert!(
        (avg_k - rounded).abs() <= f32::EPSILON,
        "benchmark `{benchmark_name}` has invalid avg_k={avg_k}; avg_k >= 1 must be an integer repeat count"
    );

    rounded as usize
}

fn compute_ratio_sample_size(total_len: usize, ratio: f32) -> usize {
    (((total_len as f64) * f64::from(ratio)).round() as usize).clamp(1, total_len)
}

fn deterministic_sample_indices(total_len: usize, sample_size: usize, seed: u64) -> Vec<usize> {
    assert!(
        sample_size <= total_len,
        "sample_size={sample_size} exceeds total_len={total_len}"
    );

    let mut indices = (0..total_len).collect::<Vec<_>>();
    let mut rng = SplitMix64::new(seed);

    for start in 0..sample_size {
        let remaining = total_len - start;
        let offset = (rng.next_u64() % remaining as u64) as usize;
        indices.swap(start, start + offset);
    }

    indices.truncate(sample_size);
    indices.sort_unstable();
    indices
}

fn fnv1a_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3_u64);
    }
    hash
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn prepare_benchmark(benchmark_info: &BenchmarkInfo, benchmark: &mut dyn Benchmark) {
    let load_invalid = benchmark.load();
    let check_invalid = if load_invalid {
        true
    } else {
        benchmark.check()
    };

    if load_invalid || check_invalid {
        // 预留逻辑: download 失败后这里会补 3 次重试。
        benchmark.download();

        let load_invalid = benchmark.load();
        let check_invalid = if load_invalid {
            true
        } else {
            benchmark.check()
        };

        assert!(
            !load_invalid && !check_invalid,
            "benchmark `{}` is still invalid after download",
            benchmark_info.name.0,
        );
    }
}
