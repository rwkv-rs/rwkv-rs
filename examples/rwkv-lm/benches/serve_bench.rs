use clap::Parser;
use rwkv_bench::Result;
use rwkv_bench::report::generate_serve_report;
use rwkv_bench::serving::{run_serve_benchmark, write_json_file};

#[path = "shared/common.rs"]
mod common;

#[derive(Parser, Debug)]
#[command(name = "serve-bench", about = "Serve pressure benchmark")]
struct Cli {
    #[command(flatten)]
    serve: common::ServeArgs,
}

fn run(cli: Cli) -> Result<()> {
    let output_json = cli
        .serve
        .output_json
        .clone()
        .unwrap_or_else(|| common::default_output_path("serve.json"));
    let report_dir = cli
        .serve
        .report_dir
        .clone()
        .unwrap_or_else(|| common::default_output_path("serve-report"));

    let runtime = common::build_runtime()?;
    let run = runtime.block_on(run_serve_benchmark(cli.serve.into()))?;
    write_json_file(&output_json, &run)?;

    let artifacts = generate_serve_report(&run, &report_dir)?;
    println!("serve json: {}", output_json.display());
    println!("report markdown: {}", artifacts.markdown.display());

    Ok(())
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("serve-bench error: {err}");
        std::process::exit(1);
    }
}
