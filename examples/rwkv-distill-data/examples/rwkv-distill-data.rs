use std::{path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use rwkv_distill_data::{paths, pipeline::synthesize};

#[derive(Parser)]
#[command(author, version, about = "RWKV distillation data pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Synthesize {
        #[arg(long, default_value_os_t = paths::default_config_path())]
        config: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Synthesize { config, limit } => synthesize(&config, limit).await,
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::FAILURE
        }
    }
}
