use std::path::PathBuf;

use clap::Parser;
use rwkv_bench::Result;
use rwkv_bench::report::{generate_report_from_input, load_report_input};

#[path = "shared/common.rs"]
mod common;

#[derive(Parser, Debug)]
#[command(name = "report-bench", about = "Generate report from benchmark JSON")]
struct Cli {
    #[arg(long)]
    input_json: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

fn run(cli: Cli) -> Result<()> {
    let output_dir = cli
        .output_dir
        .clone()
        .unwrap_or_else(|| common::default_output_path("report"));

    let input = load_report_input(&cli.input_json)?;
    let artifacts = generate_report_from_input(&input, &output_dir)?;

    println!("report markdown: {}", artifacts.markdown.display());
    for chart in artifacts.charts {
        println!("chart: {}", chart.display());
    }

    Ok(())
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("report-bench error: {err}");
        std::process::exit(1);
    }
}
