use std::path::PathBuf;

use clap::Parser;
use rwkv_bench::Result;
use rwkv_bench::profile::{ensure_success, run_external_command};
use rwkv_bench::serving::write_json_file;

#[path = "shared/common.rs"]
mod common;

#[derive(Parser, Debug)]
#[command(
    name = "train-command-bench",
    about = "Run and record a train benchmark command"
)]
struct Cli {
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<String>,
}

fn run(cli: Cli) -> Result<()> {
    let output_json = cli
        .output_json
        .clone()
        .unwrap_or_else(|| common::default_output_path("train_run.json"));

    let run = run_external_command(&cli.command)?;
    write_json_file(&output_json, &run)?;
    println!("train command json: {}", output_json.display());
    ensure_success(&run)
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("train-command-bench error: {err}");
        std::process::exit(1);
    }
}
