use std::path::PathBuf;

use clap::Parser;
use rwkv_bench::Result;
use rwkv_bench::profile::{NsysProfileArgs, ensure_success, run_nsys};

#[path = "shared/common.rs"]
mod common;

#[derive(Parser, Debug)]
#[command(
    name = "profile-nsys-bench",
    about = "Wrap a command with nsys profile"
)]
struct Cli {
    #[arg(long)]
    output_prefix: Option<PathBuf>,
    #[arg(long, default_value = "cuda,nvtx,osrt")]
    trace: String,
    #[arg(long, default_value = "none")]
    sample: String,
    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<String>,
}

fn run(cli: Cli) -> Result<()> {
    let output_prefix = cli
        .output_prefix
        .clone()
        .unwrap_or_else(|| common::default_output_path("nsys/rwkv"));

    let run = run_nsys(&NsysProfileArgs {
        output_prefix: output_prefix.clone(),
        trace: cli.trace,
        sample: cli.sample,
        command: cli.command,
    })?;

    println!("nsys output prefix: {}", output_prefix.display());
    ensure_success(&run)
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("profile-nsys-bench error: {err}");
        std::process::exit(1);
    }
}
