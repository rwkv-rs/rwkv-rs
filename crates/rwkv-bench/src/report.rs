use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use plotters::prelude::*;

use crate::serving::{ServeRunResult, SweepCaseResult, SweepRunResult};
use crate::{BenchError, Result};

#[derive(Clone, Debug)]
pub struct ReportArtifacts {
    pub markdown: PathBuf,
    pub charts: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub enum ReportInput {
    Serve(ServeRunResult),
    Sweep(SweepRunResult),
}

fn ensure_report_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

fn cdf_points(mut values: Vec<f64>) -> Vec<(f64, f64)> {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len().max(1) as f64;
    values
        .into_iter()
        .enumerate()
        .map(|(i, value)| (value, ((i + 1) as f64 / n) * 100.0))
        .collect()
}

fn write_latency_cdf_svg(run: &ServeRunResult, path: &Path) -> Result<()> {
    let ttft = run
        .requests
        .iter()
        .filter_map(|r| r.ttft_s)
        .filter(|v| *v > 0.0)
        .map(|s| s * 1000.0)
        .collect::<Vec<_>>();
    let e2el = run
        .requests
        .iter()
        .map(|r| r.e2el_s * 1000.0)
        .filter(|v| *v > 0.0)
        .collect::<Vec<_>>();

    let ttft_points = cdf_points(ttft);
    let e2el_points = cdf_points(e2el);
    let x_max = ttft_points
        .iter()
        .chain(e2el_points.iter())
        .map(|(x, _)| *x)
        .fold(1.0, f64::max)
        * 1.05;

    let root = SVGBackend::new(path, (1200, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latency CDF", ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f64..x_max, 0.0f64..100.0f64)?;

    chart
        .configure_mesh()
        .x_desc("Latency (ms)")
        .y_desc("Percentile (%)")
        .draw()?;

    if !ttft_points.is_empty() {
        chart
            .draw_series(LineSeries::new(ttft_points, BLUE.stroke_width(3)))?
            .label("TTFT")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    }

    if !e2el_points.is_empty() {
        chart
            .draw_series(LineSeries::new(e2el_points, RED.stroke_width(3)))?
            .label("E2EL")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}

fn write_throughput_svg(run: &ServeRunResult, path: &Path) -> Result<()> {
    let labels = ["req/s", "out tok/s", "total tok/s"];
    let values = [
        run.summary.request_throughput,
        run.summary.output_token_throughput,
        run.summary.total_token_throughput,
    ];
    let y_max = values
        .iter()
        .copied()
        .fold(1.0f64, f64::max)
        .mul_add(0.2, values.iter().copied().fold(1.0f64, f64::max));

    let root = SVGBackend::new(path, (1000, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Throughput Overview", ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0i32..3i32, 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_labels(3)
        .x_label_formatter(&|v| {
            let idx = (*v as usize).min(labels.len() - 1);
            labels[idx].to_string()
        })
        .y_desc("Throughput")
        .draw()?;

    for (idx, value) in values.iter().enumerate() {
        let x0 = idx as i32;
        let x1 = x0 + 1;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, *value)],
            BLUE.mix(0.6).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn write_e2el_histogram_svg(run: &ServeRunResult, path: &Path) -> Result<()> {
    let mut values = run
        .requests
        .iter()
        .map(|r| r.e2el_s * 1000.0)
        .filter(|v| *v > 0.0)
        .collect::<Vec<_>>();

    values.sort_by(|a, b| a.total_cmp(b));

    let root = SVGBackend::new(path, (1200, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    if values.is_empty() {
        root.present()?;
        return Ok(());
    }

    let min = values.first().copied().unwrap_or(0.0);
    let max = values.last().copied().unwrap_or(1.0).max(min + 1e-6);
    let bins = 20usize;
    let width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for value in values {
        let mut idx = ((value - min) / width) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }

    let max_count = counts.iter().copied().max().unwrap_or(1) as i32;

    let mut chart = ChartBuilder::on(&root)
        .caption("E2EL Histogram", ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0i32..max_count.max(1))?;

    chart
        .configure_mesh()
        .x_desc("E2EL (ms)")
        .y_desc("Count")
        .draw()?;

    for (idx, count) in counts.into_iter().enumerate() {
        let x0 = min + idx as f64 * width;
        let x1 = x0 + width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0), (x1, count as i32)],
            RED.mix(0.5).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn write_heatmap_svg(backend: &str, cases: &[&SweepCaseResult], path: &Path) -> Result<()> {
    let batch_values = cases
        .iter()
        .map(|c| c.batch_size)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let paragraph_values = cases
        .iter()
        .map(|c| c.paragraph_len)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut value_map = BTreeMap::new();
    for case in cases {
        value_map.insert(
            (case.batch_size, case.paragraph_len),
            case.run.summary.total_token_throughput,
        );
    }

    let min_v = value_map.values().copied().fold(f64::INFINITY, f64::min);
    let max_v = value_map
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(min_v + 1e-9);

    let root = SVGBackend::new(path, (1200, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sweep Heatmap ({backend}) - Total Token Throughput"),
            ("sans-serif", 32),
        )
        .margin(20)
        .x_label_area_size(70)
        .y_label_area_size(70)
        .build_cartesian_2d(
            0i32..batch_values.len() as i32,
            0i32..paragraph_values.len() as i32,
        )?;

    chart
        .configure_mesh()
        .x_labels(batch_values.len())
        .x_label_formatter(&|idx| {
            let i = (*idx as usize).min(batch_values.len().saturating_sub(1));
            format!("{}", batch_values[i])
        })
        .y_labels(paragraph_values.len())
        .y_label_formatter(&|idx| {
            let i = (*idx as usize).min(paragraph_values.len().saturating_sub(1));
            format!("{}", paragraph_values[i])
        })
        .x_desc("batch_size")
        .y_desc("paragraph_len")
        .draw()?;

    for (x_idx, batch) in batch_values.iter().enumerate() {
        for (y_idx, paragraph) in paragraph_values.iter().enumerate() {
            let value = value_map
                .get(&(*batch, *paragraph))
                .copied()
                .unwrap_or_default();
            let normalized = if (max_v - min_v).abs() < f64::EPSILON {
                0.5
            } else {
                (value - min_v) / (max_v - min_v)
            };

            let hue = 240.0 - 240.0 * normalized;
            let color = HSLColor(hue / 360.0, 0.7, 0.5);

            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (x_idx as i32, y_idx as i32),
                    (x_idx as i32 + 1, y_idx as i32 + 1),
                ],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

pub fn generate_serve_report(run: &ServeRunResult, report_dir: &Path) -> Result<ReportArtifacts> {
    ensure_report_dir(report_dir)?;

    let latency_cdf = report_dir.join("latency_cdf.svg");
    let throughput = report_dir.join("throughput.svg");
    let e2el_hist = report_dir.join("e2el_histogram.svg");
    let markdown = report_dir.join("report.md");

    write_latency_cdf_svg(run, &latency_cdf)?;
    write_throughput_svg(run, &throughput)?;
    write_e2el_histogram_svg(run, &e2el_hist)?;

    let body = format!(
        "# RWKV Serve Benchmark Report\n\n- Started: {}\n- Finished: {}\n- Duration: {:.3}s\n- Completed: {}\n- Failed: {}\n\n## Throughput\n- Request throughput: {:.3} req/s\n- Output token throughput: {:.3} tok/s\n- Total token throughput: {:.3} tok/s\n\n## Latency (ms)\n- TTFT p50/p95/p99: {:.3} / {:.3} / {:.3}\n- E2EL p50/p95/p99: {:.3} / {:.3} / {:.3}\n\n## Artifacts\n- latency_cdf.svg\n- throughput.svg\n- e2el_histogram.svg\n",
        run.started_at_utc,
        run.finished_at_utc,
        run.duration_s,
        run.summary.completed,
        run.summary.failed,
        run.summary.request_throughput,
        run.summary.output_token_throughput,
        run.summary.total_token_throughput,
        run.summary.ttft_ms.p50,
        run.summary.ttft_ms.p95,
        run.summary.ttft_ms.p99,
        run.summary.e2el_ms.p50,
        run.summary.e2el_ms.p95,
        run.summary.e2el_ms.p99,
    );

    std::fs::write(&markdown, body)?;

    Ok(ReportArtifacts {
        markdown,
        charts: vec![latency_cdf, throughput, e2el_hist],
    })
}

pub fn generate_sweep_report(run: &SweepRunResult, report_dir: &Path) -> Result<ReportArtifacts> {
    ensure_report_dir(report_dir)?;

    let markdown = report_dir.join("report.md");
    let mut charts = Vec::new();

    let mut grouped: BTreeMap<&str, Vec<&SweepCaseResult>> = BTreeMap::new();
    for case in &run.cases {
        grouped.entry(&case.backend).or_default().push(case);
    }

    for (backend, cases) in &grouped {
        let heatmap = report_dir.join(format!("sweep_heatmap_{}.svg", backend));
        write_heatmap_svg(backend, cases, &heatmap)?;
        charts.push(heatmap);
    }

    let best = run.best_case.as_ref().map_or_else(
        || "- None".to_string(),
        |case| {
            format!(
                "- backend={} batch_size={} paragraph_len={} total_tok/s={:.3} req/s={:.3}",
                case.backend,
                case.batch_size,
                case.paragraph_len,
                case.total_token_throughput,
                case.request_throughput
            )
        },
    );

    let mut body = String::new();
    body.push_str("# RWKV Sweep Report\n\n");
    body.push_str(&format!("- Started: {}\n", run.started_at_utc));
    body.push_str(&format!("- Finished: {}\n", run.finished_at_utc));
    body.push_str(&format!("- Cases: {}\n\n", run.cases.len()));
    body.push_str("## Best Case\n");
    body.push_str(&best);
    body.push_str("\n\n## Artifacts\n");
    for chart in &charts {
        if let Some(name) = chart.file_name().and_then(|n| n.to_str()) {
            body.push_str(&format!("- {}\n", name));
        }
    }

    std::fs::write(&markdown, body)?;

    Ok(ReportArtifacts { markdown, charts })
}

pub fn load_report_input(path: &Path) -> Result<ReportInput> {
    let raw = std::fs::read_to_string(path)?;

    if let Ok(run) = serde_json::from_str::<SweepRunResult>(&raw) {
        return Ok(ReportInput::Sweep(run));
    }

    if let Ok(run) = serde_json::from_str::<ServeRunResult>(&raw) {
        return Ok(ReportInput::Serve(run));
    }

    Err(BenchError::report_input_decode(path))
}

pub fn generate_report_from_input(
    input: &ReportInput,
    report_dir: &Path,
) -> Result<ReportArtifacts> {
    match input {
        ReportInput::Serve(run) => generate_serve_report(run, report_dir),
        ReportInput::Sweep(run) => generate_sweep_report(run, report_dir),
    }
}
