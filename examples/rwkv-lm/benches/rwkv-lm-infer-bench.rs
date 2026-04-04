#![recursion_limit = "256"]

#[cfg(not(feature = "inferring"))]
fn main() {
    eprintln!(
        "This bench requires feature `inferring`.\nRun: cargo bench -p rwkv-lm --bench \
         rwkv-lm-infer-bench --features cuda -- --infer-cfg rwkv-7.2b-g1e"
    );
}

#[cfg(feature = "inferring")]
mod bench {
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        fs,
        path::{Path, PathBuf},
        sync::{
            Arc,
            Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use kuva::prelude::*;
    use rwkv::{
        config::{
            get_arg_value,
            raw::{
                infer::{GenerationConfig, RawInferConfig},
                model::RawModelConfig,
            },
            validated::model::{FinalModelConfig, FinalModelConfigBuilder},
        },
        custom::{
            cubecl::device::DeviceId,
            prelude::{Backend, DeviceOps},
            store::{BurnpackStore, ModuleSnapshot},
        },
        data::tokenizer::Tokenizer,
        infer::{
            cores::queue::{QueueEvent, queue_worker::spawn_queue_worker},
            dtos::{
                completions::CompletionsReq,
                health::{
                    GpuSample,
                    HealthResp,
                    QueueHealthBinding,
                    QueuePerfSample,
                    QueuePerfStage,
                },
            },
            sdk::LocalClient,
            services::{QueueMap, ServiceError, ServiceResult},
        },
        nn::kernels::{
            addcmul::AddcmulBackend,
            rapid_sample::RapidSampleBackend,
            token_shift_diff::TokenShiftDiffBackend,
            wkv7_common::Wkv7Backend,
        },
    };
    use serde::de::DeserializeOwned;
    use tokio::task::JoinSet;
    use rwkv::nn::kernels::guided_token_mask::GuidedTokenMaskBackend;
    use rwkv_lm::{
        inferring::{RwkvLmForward, infer_cli_args},
        model::AutoRegressiveModelConfig,
        paths,
    };

    // Keep the prompt short and the decode budget long so this bench stresses token-by-token
    // inference instead of spending most of its time in prefill.
    const BENCH_PROMPT: &str = "User: Give a long, detailed explanation of field extensions, with many examples.\nAssistant:";
    const BENCH_MAX_TOKENS: u32 = 512;

    #[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
    pub type ElemType = rwkv::custom::tensor::bf16;
    #[cfg(feature = "f32")]
    pub type ElemType = f32;
    #[cfg(feature = "flex32")]
    pub type ElemType = rwkv::custom::tensor::flex32;
    #[cfg(feature = "f16")]
    pub type ElemType = rwkv::custom::tensor::f16;

    struct RequestOutcome {
        latency_ms: f64,
        generated_tokens: usize,
        error: Option<String>,
    }

    impl RequestOutcome {
        fn success(latency_ms: f64, generated_tokens: usize) -> Self {
            Self {
                latency_ms,
                generated_tokens,
                error: None,
            }
        }

        fn failed(latency_ms: f64, error: String) -> Self {
            Self {
                latency_ms,
                generated_tokens: 0,
                error: Some(error),
            }
        }
    }

    struct StageResult {
        concurrency: usize,
        elapsed_secs: f64,
        succeeded: usize,
        failed: usize,
        total_output_tokens: usize,
        mean_latency_ms: f64,
        request_throughput: f64,
        output_tokens_per_sec: f64,
        first_error: Option<String>,
        stage_start_unix_ms: u64,
        stage_end_unix_ms: u64,
        health_snapshots: Vec<HealthResp>,
    }

    struct ModelBenchResult {
        model_name: String,
        max_batch_size: usize,
        stages: Vec<StageResult>,
    }

    struct LineSeriesData {
        label: String,
        points: Vec<(f64, f64)>,
    }

    pub async fn run_backend<B>() -> Result<(), String>
    where
        B: Backend
            + TokenShiftDiffBackend
            + AddcmulBackend
            + Wkv7Backend
            + GuidedTokenMaskBackend
            + RapidSampleBackend
            + Send
            + Sync
            + 'static,
    {
        let args: Vec<String> = std::env::args().collect();
        let (config_dir, infer_cfg) = infer_cli_args(&args);
        let output_root = get_arg_value(&args, "--output-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| paths::bench_output_path("rwkv-lm-infer-bench"));
        let output_root = output_root.join(sanitize_path_component(&infer_cfg));
        fs::create_dir_all(&output_root)
            .map_err(|err| format!("failed to create {}: {err}", output_root.display()))?;

        let (client, models) = build_local_client::<B>(&config_dir, &infer_cfg)
            .map_err(|err| service_error_message(&err))?;

        println!(
            "bench infer cfg: {infer_cfg} (config_dir: {})",
            config_dir.display()
        );
        println!("bench output: {}", output_root.display());

        for model in models {
            let model_dir = output_root.join(sanitize_path_component(&model.model_name));
            fs::create_dir_all(&model_dir)
                .map_err(|err| format!("failed to create {}: {err}", model_dir.display()))?;

            let result = bench_model(client.clone(), &model).await;
            print_model_summary(&result, &model_dir);
            write_model_charts(&model_dir, &result)?;
        }

        Ok(())
    }

    async fn bench_model(client: LocalClient, model: &GenerationConfig) -> ModelBenchResult {
        let max_batch_size = model.max_batch_size.unwrap_or(1);
        let levels = concurrency_levels(max_batch_size);
        println!(
            "\nmodel={} max_batch_size={} levels={levels:?}",
            model.model_name, max_batch_size
        );

        let mut stages = Vec::with_capacity(levels.len());
        for concurrency in levels {
            let stage = run_stage(client.clone(), &model.model_name, concurrency).await;
            print_stage_summary(&model.model_name, &stage);
            stages.push(stage);
        }

        ModelBenchResult {
            model_name: model.model_name.clone(),
            max_batch_size,
            stages,
        }
    }

    async fn run_stage(client: LocalClient, model_name: &str, concurrency: usize) -> StageResult {
        let stage_start_unix_ms = unix_millis_now();
        let stage_started = Instant::now();
        let stop_sampling = Arc::new(AtomicBool::new(false));
        let health_snapshots = Arc::new(Mutex::new(Vec::new()));
        let health_sampling_interval_ms = health_sampling_interval_ms();

        {
            let mut snapshots = health_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            snapshots.push(client.health());
        }

        let sampler = (health_sampling_interval_ms > 0).then(|| {
            let client = client.clone();
            let stop_sampling = Arc::clone(&stop_sampling);
            let health_snapshots = Arc::clone(&health_snapshots);
            tokio::spawn(async move {
                sample_health_loop(
                    client,
                    health_snapshots,
                    stop_sampling,
                    health_sampling_interval_ms,
                )
                .await;
            })
        });

        let template = build_completion_request(model_name);
        let mut join_set = JoinSet::new();
        for _ in 0..concurrency {
            let client = client.clone();
            let req = template.clone();
            join_set.spawn(async move { run_completion(client, req).await });
        }

        let mut outcomes = Vec::with_capacity(concurrency);
        while let Some(joined) = join_set.join_next().await {
            match joined {
                Ok(outcome) => outcomes.push(outcome),
                Err(err) => outcomes.push(RequestOutcome::failed(
                    0.0,
                    format!("request task panicked: {err}"),
                )),
            }
        }

        if let Some(sampler) = sampler {
            stop_sampling.store(true, Ordering::Relaxed);
            let _ = sampler.await;
        }

        {
            let mut snapshots = health_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            snapshots.push(client.health());
        }
        let stage_end_unix_ms = unix_millis_now();

        let elapsed_secs = stage_started.elapsed().as_secs_f64();
        let succeeded = outcomes
            .iter()
            .filter(|outcome| outcome.error.is_none())
            .count();
        let failed = outcomes.len().saturating_sub(succeeded);
        let total_output_tokens = outcomes
            .iter()
            .map(|outcome| outcome.generated_tokens)
            .sum::<usize>();
        let total_latency_ms = outcomes
            .iter()
            .map(|outcome| outcome.latency_ms)
            .sum::<f64>();
        let mean_latency_ms = if outcomes.is_empty() {
            0.0
        } else {
            total_latency_ms / outcomes.len() as f64
        };
        let request_throughput = if elapsed_secs > 0.0 {
            succeeded as f64 / elapsed_secs
        } else {
            0.0
        };
        let output_tokens_per_sec = if elapsed_secs > 0.0 {
            total_output_tokens as f64 / elapsed_secs
        } else {
            0.0
        };
        let first_error = outcomes.iter().find_map(|outcome| outcome.error.clone());
        let health_snapshots = health_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        StageResult {
            concurrency,
            elapsed_secs,
            succeeded,
            failed,
            total_output_tokens,
            mean_latency_ms,
            request_throughput,
            output_tokens_per_sec,
            first_error,
            stage_start_unix_ms,
            stage_end_unix_ms,
            health_snapshots,
        }
    }

    async fn sample_health_loop(
        client: LocalClient,
        health_snapshots: Arc<Mutex<Vec<HealthResp>>>,
        stop_sampling: Arc<AtomicBool>,
        interval_ms: u64,
    ) {
        loop {
            let snapshot = client.health();
            health_snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(snapshot);

            if stop_sampling.load(Ordering::Relaxed) {
                break;
            }

            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    }

    fn health_sampling_interval_ms() -> u64 {
        std::env::var("RWKV_LM_INFER_BENCH_HEALTH_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    }

    async fn run_completion(client: LocalClient, req: CompletionsReq) -> RequestOutcome {
        let started = Instant::now();

        match client.completions_run(req).await {
            Ok(mut run) => {
                while let Some(event) = run.rx.recv().await {
                    if let QueueEvent::Done(meta) = event {
                        return RequestOutcome::success(
                            started.elapsed().as_secs_f64() * 1000.0,
                            meta.generated_tokens,
                        );
                    }
                }

                RequestOutcome::failed(
                    started.elapsed().as_secs_f64() * 1000.0,
                    "queue stream closed without finish meta".to_string(),
                )
            }
            Err(err) => RequestOutcome::failed(
                started.elapsed().as_secs_f64() * 1000.0,
                service_error_message(&err),
            ),
        }
    }

    fn build_completion_request(model_name: &str) -> CompletionsReq {
        CompletionsReq {
            model: model_name.to_string(),
            prompt: BENCH_PROMPT.to_string(),
            stream: Some(false),
            max_tokens: Some(BENCH_MAX_TOKENS),
            temperature: Some(0.8),
            top_k: Some(50),
            top_p: Some(0.95),
            presence_penalty: Some(0.5),
            repetition_penalty: Some(0.5),
            penalty_decay: Some(0.996),
            // Disable early stop strings so stages consistently measure long decode runs instead
            // of whichever request happened to terminate first.
            stop: None,
            logprobs: None,
            candidate_token_texts: None,
        }
    }

    fn concurrency_levels(max_batch_size: usize) -> Vec<usize> {
        let mut levels = vec![1usize];
        for candidate in [
            max_batch_size / 8,
            max_batch_size / 4,
            max_batch_size / 2,
            (max_batch_size * 3) / 4,
            max_batch_size,
        ] {
            levels.push(candidate.clamp(1, max_batch_size));
        }

        levels.dedup();
        levels
    }

    fn print_stage_summary(model_name: &str, stage: &StageResult) {
        println!(
            "  model={} concurrency={} ok={} fail={} elapsed={:.3}s req/s={:.3} out_tok/s={:.3} mean_latency_ms={:.3} total_out_tokens={}",
            model_name,
            stage.concurrency,
            stage.succeeded,
            stage.failed,
            stage.elapsed_secs,
            stage.request_throughput,
            stage.output_tokens_per_sec,
            stage.mean_latency_ms,
            stage.total_output_tokens,
        );

        if let Some(error) = &stage.first_error {
            println!("    first_error={error}");
        }
    }

    fn print_model_summary(result: &ModelBenchResult, model_dir: &Path) {
        let best_stage = best_stage(result);
        println!(
            "model={} best_concurrency={} max_batch_size={} charts={}",
            result.model_name,
            best_stage.concurrency,
            result.max_batch_size,
            model_dir.display(),
        );
    }

    fn write_model_charts(model_dir: &Path, result: &ModelBenchResult) -> Result<(), String> {
        let throughput_series = vec![LineSeriesData {
            label: "request throughput".to_string(),
            points: result
                .stages
                .iter()
                .map(|stage| (stage.concurrency as f64, stage.request_throughput))
                .collect(),
        }];
        write_line_chart(
            &model_dir.join("concurrency_vs_request_throughput.svg"),
            &format!("{} Request Throughput", result.model_name),
            "Concurrency",
            "Requests / sec",
            &throughput_series,
            true,
        )?;

        let output_tokens_series = vec![LineSeriesData {
            label: "output throughput".to_string(),
            points: result
                .stages
                .iter()
                .map(|stage| (stage.concurrency as f64, stage.output_tokens_per_sec))
                .collect(),
        }];
        write_line_chart(
            &model_dir.join("concurrency_vs_output_tokens_per_sec.svg"),
            &format!("{} Output Token Throughput", result.model_name),
            "Concurrency",
            "Tokens / sec",
            &output_tokens_series,
            true,
        )?;

        let latency_series = vec![LineSeriesData {
            label: "mean latency".to_string(),
            points: result
                .stages
                .iter()
                .map(|stage| (stage.concurrency as f64, stage.mean_latency_ms))
                .collect(),
        }];
        write_line_chart(
            &model_dir.join("concurrency_vs_mean_latency_ms.svg"),
            &format!("{} Mean Latency", result.model_name),
            "Concurrency",
            "Latency (ms)",
            &latency_series,
            true,
        )?;

        let best_stage = best_stage(result);
        let prefill_series = queue_series(
            best_stage,
            &result.model_name,
            Some(QueuePerfStage::Prefill),
            |sample| sample.instant_tokens_per_sec,
        );
        write_line_chart(
            &model_dir.join("best_concurrency_queue_prefill_tps.svg"),
            &format!(
                "{} Prefill Speed at Concurrency {}",
                result.model_name, best_stage.concurrency
            ),
            "Seconds Since Stage Start",
            "Tokens / sec",
            &prefill_series,
            false,
        )?;

        let decode_series = queue_series(
            best_stage,
            &result.model_name,
            Some(QueuePerfStage::Decode),
            |sample| sample.instant_tokens_per_sec,
        );
        write_line_chart(
            &model_dir.join("best_concurrency_queue_decode_tps.svg"),
            &format!(
                "{} Decode Speed at Concurrency {}",
                result.model_name, best_stage.concurrency
            ),
            "Seconds Since Stage Start",
            "Tokens / sec",
            &decode_series,
            false,
        )?;

        let batch_utilization_series =
            queue_series(best_stage, &result.model_name, None, |sample| {
                sample.batch_utilization * 100.0
            });
        write_line_chart(
            &model_dir.join("best_concurrency_queue_batch_utilization.svg"),
            &format!(
                "{} Batch Utilization at Concurrency {}",
                result.model_name, best_stage.concurrency
            ),
            "Seconds Since Stage Start",
            "Batch Utilization (%)",
            &batch_utilization_series,
            false,
        )?;

        let gpu_utilization_series = gpu_series(best_stage, |sample| sample.utilization_percent);
        write_line_chart(
            &model_dir.join("best_concurrency_gpu_utilization.svg"),
            &format!(
                "{} GPU Utilization at Concurrency {}",
                result.model_name, best_stage.concurrency
            ),
            "Seconds Since Stage Start",
            "GPU Utilization (%)",
            &gpu_utilization_series,
            false,
        )?;

        let gpu_memory_series = gpu_series(best_stage, |sample| sample.memory_utilization_percent);
        write_line_chart(
            &model_dir.join("best_concurrency_gpu_memory_utilization.svg"),
            &format!(
                "{} GPU Memory Utilization at Concurrency {}",
                result.model_name, best_stage.concurrency
            ),
            "Seconds Since Stage Start",
            "GPU Memory Utilization (%)",
            &gpu_memory_series,
            false,
        )?;

        Ok(())
    }

    fn best_stage(result: &ModelBenchResult) -> &StageResult {
        result
            .stages
            .iter()
            .max_by(|left, right| {
                left.request_throughput
                    .total_cmp(&right.request_throughput)
                    .then(left.concurrency.cmp(&right.concurrency))
            })
            .expect("at least one stage result")
    }

    fn queue_series<F>(
        stage: &StageResult,
        model_name: &str,
        wanted_stage: Option<QueuePerfStage>,
        value_fn: F,
    ) -> Vec<LineSeriesData>
    where
        F: Fn(&QueuePerfSample) -> f64,
    {
        let mut series_map: BTreeMap<String, Vec<QueuePerfSample>> = BTreeMap::new();

        for snapshot in &stage.health_snapshots {
            for panel in &snapshot.gpu_panels {
                for queue in &panel.queues {
                    if queue.model_name != model_name {
                        continue;
                    }

                    let label = format_queue_label(queue);
                    let entry = series_map.entry(label).or_default();
                    entry.extend(
                        queue
                            .samples
                            .iter()
                            .filter(|sample| {
                                sample.ts_unix_ms >= stage.stage_start_unix_ms
                                    && sample.ts_unix_ms <= stage.stage_end_unix_ms
                                    && wanted_stage
                                        .map_or(true, |stage_filter| sample.stage == stage_filter)
                            })
                            .cloned(),
                    );
                }
            }
        }

        series_map
            .into_iter()
            .map(|(label, mut samples)| {
                sort_and_dedup_queue_samples(&mut samples);
                let points = samples
                    .into_iter()
                    .map(|sample| {
                        (
                            sample.ts_unix_ms.saturating_sub(stage.stage_start_unix_ms) as f64
                                / 1000.0,
                            value_fn(&sample),
                        )
                    })
                    .collect();
                LineSeriesData { label, points }
            })
            .collect()
    }

    fn gpu_series<F>(stage: &StageResult, value_fn: F) -> Vec<LineSeriesData>
    where
        F: Fn(&GpuSample) -> Option<f64>,
    {
        let mut series_map: BTreeMap<String, Vec<GpuSample>> = BTreeMap::new();

        for snapshot in &stage.health_snapshots {
            for gpu in &snapshot.gpu_panels {
                let label = format!("device{} {} ({})", gpu.device_id, gpu.name, gpu.device_key);
                let entry = series_map.entry(label).or_default();
                entry.extend(
                    gpu.samples
                        .iter()
                        .filter(|sample| {
                            sample.ts_unix_ms >= stage.stage_start_unix_ms
                                && sample.ts_unix_ms <= stage.stage_end_unix_ms
                        })
                        .cloned(),
                );
            }
        }

        series_map
            .into_iter()
            .map(|(label, mut samples)| {
                sort_and_dedup_gpu_samples(&mut samples);
                let points = samples
                    .into_iter()
                    .filter_map(|sample| {
                        value_fn(&sample).map(|value| {
                            (
                                sample.ts_unix_ms.saturating_sub(stage.stage_start_unix_ms) as f64
                                    / 1000.0,
                                value,
                            )
                        })
                    })
                    .collect();
                LineSeriesData { label, points }
            })
            .collect()
    }

    fn write_line_chart(
        path: &Path,
        title: &str,
        x_label: &str,
        y_label: &str,
        series: &[LineSeriesData],
        integer_x_axis: bool,
    ) -> Result<(), String> {
        let palette = Palette::tol_bright();
        let mut plots = Vec::new();

        for (index, series) in series
            .iter()
            .filter(|series| !series.points.is_empty())
            .enumerate()
        {
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(series.points.iter().copied())
                    .with_color(palette[index].to_string())
                    .with_stroke_width(2.5)
                    .with_legend(series.label.clone()),
            ));
        }

        let had_data = !plots.is_empty();
        if !had_data {
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(vec![(0.0_f64, 0.0_f64)])
                    .with_color("#999999")
                    .with_stroke_width(2.0)
                    .with_legend("no data"),
            ));
        }

        let mut layout = Layout::auto_from_plots(&plots)
            .with_title(if had_data {
                title.to_string()
            } else {
                format!("{title} (no data)")
            })
            .with_x_label(x_label)
            .with_y_label(y_label)
            .with_width(1280.0)
            .with_height(720.0)
            .with_palette(Palette::tol_bright())
            .with_legend_position(LegendPosition::OutsideRightTop);

        if integer_x_axis {
            layout = layout.with_x_tick_format(TickFormat::Integer);
        }

        let svg = render_to_svg(plots, layout);
        fs::write(path, svg).map_err(|err| format!("failed to write {}: {err}", path.display()))
    }

    fn format_queue_label(queue: &QueueHealthBinding) -> String {
        let weights_name = Path::new(&queue.weights_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(queue.weights_path.as_str());
        format!("device{} {}", queue.device_id, weights_name)
    }

    fn sort_and_dedup_queue_samples(samples: &mut Vec<QueuePerfSample>) {
        samples.sort_by(|left, right| {
            left.ts_unix_ms
                .cmp(&right.ts_unix_ms)
                .then(queue_perf_stage_order(left.stage).cmp(&queue_perf_stage_order(right.stage)))
                .then(left.duration_ms.total_cmp(&right.duration_ms))
                .then(
                    left.instant_tokens_per_sec
                        .total_cmp(&right.instant_tokens_per_sec),
                )
                .then(left.batch_used.cmp(&right.batch_used))
                .then(left.max_batch_size.cmp(&right.max_batch_size))
                .then(left.batch_utilization.total_cmp(&right.batch_utilization))
        });
        samples.dedup_by(|left, right| {
            left.ts_unix_ms == right.ts_unix_ms
                && left.stage == right.stage
                && left.duration_ms.to_bits() == right.duration_ms.to_bits()
                && left.instant_tokens_per_sec.to_bits() == right.instant_tokens_per_sec.to_bits()
                && left.batch_used == right.batch_used
                && left.max_batch_size == right.max_batch_size
                && left.batch_utilization.to_bits() == right.batch_utilization.to_bits()
        });
    }

    fn sort_and_dedup_gpu_samples(samples: &mut Vec<GpuSample>) {
        samples.sort_by(|left, right| {
            left.ts_unix_ms
                .cmp(&right.ts_unix_ms)
                .then(option_f64_total_cmp(
                    left.utilization_percent,
                    right.utilization_percent,
                ))
                .then(left.memory_used_bytes.cmp(&right.memory_used_bytes))
                .then(left.memory_total_bytes.cmp(&right.memory_total_bytes))
                .then(option_f64_total_cmp(
                    left.memory_utilization_percent,
                    right.memory_utilization_percent,
                ))
                .then(gpu_status_order(left.status).cmp(&gpu_status_order(right.status)))
        });
        samples.dedup_by(|left, right| {
            left.ts_unix_ms == right.ts_unix_ms
                && option_f64_to_bits(left.utilization_percent)
                    == option_f64_to_bits(right.utilization_percent)
                && left.memory_used_bytes == right.memory_used_bytes
                && left.memory_total_bytes == right.memory_total_bytes
                && option_f64_to_bits(left.memory_utilization_percent)
                    == option_f64_to_bits(right.memory_utilization_percent)
                && left.status == right.status
        });
    }

    fn option_f64_total_cmp(left: Option<f64>, right: Option<f64>) -> std::cmp::Ordering {
        match (left, right) {
            (Some(left), Some(right)) => left.total_cmp(&right),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    }

    fn option_f64_to_bits(value: Option<f64>) -> Option<u64> {
        value.map(f64::to_bits)
    }

    fn queue_perf_stage_order(stage: QueuePerfStage) -> u8 {
        match stage {
            QueuePerfStage::Prefill => 0,
            QueuePerfStage::Decode => 1,
        }
    }

    fn gpu_status_order(status: rwkv::infer::dtos::health::GpuSampleStatus) -> u8 {
        match status {
            rwkv::infer::dtos::health::GpuSampleStatus::Ok => 0,
            rwkv::infer::dtos::health::GpuSampleStatus::Unavailable => 1,
        }
    }

    fn unix_millis_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis() as u64
    }

    fn sanitize_path_component(value: &str) -> String {
        let sanitized = value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>();

        if sanitized.is_empty() {
            "bench".to_string()
        } else {
            sanitized
        }
    }

    fn service_error_message(err: &ServiceError) -> String {
        format!("{} {}", err.status_code(), err.body().error.message)
    }

    fn build_local_client<B>(
        config_dir: &Path,
        infer_cfg_name: &str,
    ) -> ServiceResult<(LocalClient, Vec<GenerationConfig>)>
    where
        B: Backend
            + TokenShiftDiffBackend
            + AddcmulBackend
            + Wkv7Backend
            + GuidedTokenMaskBackend
            + RapidSampleBackend
            + Send
            + Sync
            + 'static,
    {
        let infer_cfg_path = config_dir
            .join("infer")
            .join(format!("{infer_cfg_name}.toml"));
        let infer_cfg = load_raw_infer_cfg(&infer_cfg_path)?;
        let queues = build_queue_map_from_generation_cfgs::<B>(
            config_dir,
            &infer_cfg_path,
            &infer_cfg.models,
        )?;
        Ok((LocalClient::new(queues), infer_cfg.models))
    }

    fn build_queue_map_from_generation_cfgs<B>(
        config_dir: &Path,
        infer_cfg_path: &Path,
        models: &[GenerationConfig],
    ) -> ServiceResult<QueueMap>
    where
        B: Backend
            + TokenShiftDiffBackend
            + AddcmulBackend
            + Wkv7Backend
            + GuidedTokenMaskBackend
            + RapidSampleBackend
            + Send
            + Sync
            + 'static,
    {
        validate_generation_models(models)?;
        let infer_cfg_dir = infer_cfg_path.parent().unwrap_or_else(|| Path::new("."));
        let runtime_models = resolve_runtime_models(infer_cfg_dir, models);
        let model_cfgs = load_model_cfgs(config_dir, infer_cfg_dir, &runtime_models)?;
        Ok(build_queue_map::<B>(&runtime_models, &model_cfgs))
    }

    fn build_queue_map<B>(
        models: &[GenerationConfig],
        model_cfgs: &HashMap<String, Arc<FinalModelConfig>>,
    ) -> QueueMap
    where
        B: Backend
            + TokenShiftDiffBackend
            + AddcmulBackend
            + Wkv7Backend
            + GuidedTokenMaskBackend
            + RapidSampleBackend
            + Send
            + Sync
            + 'static,
    {
        let mut queue_map: QueueMap = HashMap::new();

        for generation_cfg in models {
            let model_cfg = model_cfgs.get(&generation_cfg.model_name).unwrap();
            let device_type = generation_cfg.device_type.unwrap();
            let max_batch_size = generation_cfg.max_batch_size.unwrap();
            let paragraph_len = generation_cfg.paragraph_len.unwrap();
            let tokenizer = Arc::new(Tokenizer::new(&generation_cfg.tokenizer_vocab_path).unwrap());

            for &device_id in &generation_cfg.device_ids {
                let device = B::Device::from_id(DeviceId::new(device_type, device_id));
                let model_config = AutoRegressiveModelConfig::new(
                    model_cfg.num_cells,
                    model_cfg.vocab_size,
                    model_cfg.embedded_dim,
                    model_cfg.num_heads,
                    model_cfg.head_size_auto,
                );
                let mut model_runtime = model_config.init::<B>(&device);
                let mut store =
                    BurnpackStore::from_file(&generation_cfg.weights_path).zero_copy(true);
                model_runtime.load_from(&mut store).unwrap();
                let executor = RwkvLmForward::<B>::new(
                    model_runtime,
                    model_cfg.clone(),
                    max_batch_size,
                    device.clone(),
                );
                let handle = spawn_queue_worker(
                    Box::new(executor),
                    Arc::clone(&tokenizer),
                    max_batch_size,
                    paragraph_len,
                    device_id,
                    generation_cfg.weights_path.clone(),
                );

                queue_map
                    .entry(generation_cfg.model_name.clone())
                    .or_default()
                    .push(handle);
            }
        }

        queue_map
    }

    fn load_raw_infer_cfg(path: &Path) -> ServiceResult<RawInferConfig> {
        let mut cfg: RawInferConfig = read_toml_file(path)?;
        cfg.fill_default();
        validate_generation_models(&cfg.models)?;
        Ok(cfg)
    }

    fn read_toml_file<T: DeserializeOwned>(path: &Path) -> ServiceResult<T> {
        let content = fs::read_to_string(path).map_err(|err| {
            ServiceError::bad_request(format!("failed to read {}: {err}", path.display()))
        })?;
        toml::from_str(&content).map_err(|err| {
            ServiceError::bad_request(format!("invalid toml {}: {err}", path.display()))
        })
    }

    fn validate_generation_models(models: &[GenerationConfig]) -> ServiceResult<()> {
        if models.is_empty() {
            return Err(ServiceError::bad_request(
                "infer config requires at least one model",
            ));
        }

        let mut names = HashSet::new();
        for model in models {
            if model.model_name.trim().is_empty() {
                return Err(ServiceError::bad_request("model_name cannot be empty"));
            }
            if !names.insert(model.model_name.clone()) {
                return Err(ServiceError::bad_request(format!(
                    "duplicated model_name: {}",
                    model.model_name
                )));
            }
            if model.model_cfg.trim().is_empty() {
                return Err(ServiceError::bad_request(format!(
                    "model_cfg cannot be empty for model {}",
                    model.model_name
                )));
            }
            if model.weights_path.trim().is_empty() {
                return Err(ServiceError::bad_request(format!(
                    "weights_path cannot be empty for model {}",
                    model.model_name
                )));
            }
            match Path::new(&model.weights_path)
                .extension()
                .and_then(|extension| extension.to_str())
            {
                Some("bpk") => {}
                Some("mpk") => {
                    return Err(ServiceError::bad_request(format!(
                        "weights_path for model {} points to unsupported .mpk file {}; convert it to .bpk first",
                        model.model_name, model.weights_path
                    )));
                }
                _ => {
                    return Err(ServiceError::bad_request(format!(
                        "weights_path for model {} must point to a .bpk file, got {}",
                        model.model_name, model.weights_path
                    )));
                }
            }
            if model.tokenizer_vocab_path.trim().is_empty() {
                return Err(ServiceError::bad_request(format!(
                    "tokenizer_vocab_path cannot be empty for model {}",
                    model.model_name
                )));
            }
            if model.device_ids.is_empty() {
                return Err(ServiceError::bad_request(format!(
                    "device_ids cannot be empty for model {}",
                    model.model_name
                )));
            }
            if model.max_batch_size.unwrap_or_default() < 1 {
                return Err(ServiceError::bad_request(format!(
                    "max_batch_size must be >= 1 for model {}",
                    model.model_name
                )));
            }
            if model.max_context_len.unwrap_or_default() < 1 {
                return Err(ServiceError::bad_request(format!(
                    "max_context_len must be >= 1 for model {}",
                    model.model_name
                )));
            }
        }

        Ok(())
    }

    fn load_model_cfgs(
        config_dir: &Path,
        infer_cfg_dir: &Path,
        models: &[GenerationConfig],
    ) -> ServiceResult<HashMap<String, Arc<FinalModelConfig>>> {
        let mut model_cfgs = HashMap::new();

        for model in models {
            let model_cfg_path =
                resolve_model_cfg_path(config_dir, infer_cfg_dir, &model.model_cfg);
            let mut raw_model_cfg: RawModelConfig = read_toml_file(&model_cfg_path)?;
            raw_model_cfg.fill_default();

            let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
            model_cfg_builder.fill_auto_after_load();
            model_cfgs.insert(model.model_name.clone(), model_cfg_builder.build_local());
        }

        Ok(model_cfgs)
    }

    fn resolve_runtime_models(
        infer_cfg_dir: &Path,
        models: &[GenerationConfig],
    ) -> Vec<GenerationConfig> {
        let mut resolved_models = models.to_vec();
        for model in &mut resolved_models {
            model.weights_path = resolve_path(infer_cfg_dir, &model.weights_path);
            model.tokenizer_vocab_path = resolve_path(infer_cfg_dir, &model.tokenizer_vocab_path);
        }
        resolved_models
    }

    fn resolve_model_cfg_path(config_dir: &Path, infer_cfg_dir: &Path, model_cfg: &str) -> PathBuf {
        if model_cfg.contains('/') || model_cfg.contains('\\') {
            let path = PathBuf::from(model_cfg);
            if path.is_absolute() {
                path
            } else {
                infer_cfg_dir.join(path)
            }
        } else {
            config_dir.join("model").join(format!("{model_cfg}.toml"))
        }
    }

    fn resolve_path(base_dir: &Path, path: &str) -> String {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            path.to_string()
        } else {
            base_dir.join(candidate).to_string_lossy().to_string()
        }
    }
}

#[cfg(feature = "inferring")]
#[cfg(feature = "wgpu")]
mod wgpu {
    use rwkv::custom::backend::Wgpu;

    use super::bench::{ElemType, run_backend};

    pub async fn run() -> Result<(), String> {
        run_backend::<Wgpu<ElemType, i32>>().await
    }
}

#[cfg(feature = "inferring")]
#[cfg(feature = "vulkan")]
mod vulkan {
    use rwkv::custom::backend::Vulkan;

    use super::bench::{ElemType, run_backend};

    pub async fn run() -> Result<(), String> {
        run_backend::<Vulkan<ElemType, i32>>().await
    }
}

#[cfg(feature = "inferring")]
#[cfg(feature = "cuda")]
mod cuda {
    use rwkv::custom::backend::Cuda;

    use super::bench::{ElemType, run_backend};

    pub async fn run() -> Result<(), String> {
        run_backend::<Cuda<ElemType, i32>>().await
    }
}

#[cfg(feature = "inferring")]
#[cfg(feature = "rocm")]
mod rocm {
    use rwkv::custom::backend::Hip;

    use super::bench::{ElemType, run_backend};

    pub async fn run() -> Result<(), String> {
        run_backend::<Hip<ElemType, i32>>().await
    }
}

#[cfg(feature = "inferring")]
#[cfg(feature = "metal")]
mod metal {
    use rwkv::custom::backend::Metal;

    use super::bench::{ElemType, run_backend};

    pub async fn run() -> Result<(), String> {
        run_backend::<Metal<ElemType, i32>>().await
    }
}

#[cfg(feature = "inferring")]
#[tokio::main]
async fn main() {
    #[cfg(not(any(
        feature = "wgpu",
        feature = "vulkan",
        feature = "cuda",
        feature = "rocm",
        feature = "metal"
    )))]
    {
        eprintln!(
            "This bench requires a backend feature.\nRun: cargo bench -p rwkv-lm --bench \
             rwkv-lm-infer-bench --features cuda -- --infer-cfg rwkv-7.2b-g1e"
        );
        return;
    }

    #[cfg(feature = "wgpu")]
    if let Err(err) = wgpu::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
    #[cfg(feature = "vulkan")]
    if let Err(err) = vulkan::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
    #[cfg(feature = "cuda")]
    if let Err(err) = cuda::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
    #[cfg(feature = "rocm")]
    if let Err(err) = rocm::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
    #[cfg(feature = "metal")]
    if let Err(err) = metal::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
