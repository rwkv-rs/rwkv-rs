use clap::{Args, Parser};
use rwkv_bench::Result;
use rwkv_bench::report::generate_sweep_report;
use rwkv_bench::serving::{SweepConfig, run_sweep, write_json_file};

#[path = "shared/common.rs"]
mod common;

#[derive(Args, Debug, Clone)]
struct SweepArgs {
    #[command(flatten)]
    serve: common::ServeArgs,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "64,128,256,512,1024,2048"
    )]
    request_counts: Vec<usize>,
    #[arg(long)]
    before_each_command: Option<String>,
    #[arg(long)]
    after_each_command: Option<String>,
}

#[derive(Parser, Debug)]
#[command(name = "sweep-bench", about = "Sweep benchmark matrix")]
struct Cli {
    #[command(flatten)]
    sweep: SweepArgs,
}

fn run(cli: Cli) -> Result<()> {
    let output_json = cli
        .sweep
        .serve
        .output_json
        .clone()
        .unwrap_or_else(|| common::default_output_path("sweep.json"));
    let report_dir = cli
        .sweep
        .serve
        .report_dir
        .clone()
        .unwrap_or_else(|| common::default_output_path("sweep-report"));

    let cfg = SweepConfig {
        base: cli.sweep.serve.into(),
        request_counts: cli.sweep.request_counts,
        before_each_command: cli.sweep.before_each_command,
        after_each_command: cli.sweep.after_each_command,
    };

    let runtime = common::build_runtime()?;
    let run = runtime.block_on(run_sweep(cfg))?;
    write_json_file(&output_json, &run)?;

    let artifacts = generate_sweep_report(&run, &report_dir)?;
    println!("sweep json: {}", output_json.display());
    println!("report markdown: {}", artifacts.markdown.display());

    Ok(())
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("sweep-bench error: {err}");
        std::process::exit(1);
    }
}
