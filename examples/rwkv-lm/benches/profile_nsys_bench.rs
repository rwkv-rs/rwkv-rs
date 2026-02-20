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
    #[arg(long, default_value = "logs/bench/nsys/rwkv")]
    output_prefix: PathBuf,
    #[arg(long, default_value = "cuda,nvtx,osrt")]
    trace: String,
    #[arg(long, default_value = "none")]
    sample: String,
    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<String>,
}

fn run(cli: Cli) -> Result<()> {
    let run = run_nsys(&NsysProfileArgs {
        output_prefix: cli.output_prefix,
        trace: cli.trace,
        sample: cli.sample,
        command: cli.command,
    })?;

    ensure_success(&run)
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("profile-nsys-bench error: {err}");
        std::process::exit(1);
    }
}
