use clap::Parser;
use rwkv_bench::Result;
use rwkv_bench::profile::{ensure_success, run_tracy_passthrough};

#[path = "shared/common.rs"]
mod common;

#[derive(Parser, Debug)]
#[command(
    name = "profile-tracy-bench",
    about = "Run a tracy-instrumented target command"
)]
struct Cli {
    #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
    command: Vec<String>,
}

fn run(cli: Cli) -> Result<()> {
    let run = run_tracy_passthrough(&cli.command)?;
    ensure_success(&run)
}

fn main() {
    let cli = Cli::parse_from(common::normalized_args());
    if let Err(err) = run(cli) {
        eprintln!("profile-tracy-bench error: {err}");
        std::process::exit(1);
    }
}
