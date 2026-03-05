#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Args, ValueEnum};
use rwkv_bench::serving::{Endpoint, ServeConfig};
use rwkv_lm::paths;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum EndpointArg {
    Completions,
    ChatCompletions,
}

impl From<EndpointArg> for Endpoint {
    fn from(value: EndpointArg) -> Self {
        match value {
            EndpointArg::Completions => Endpoint::Completions,
            EndpointArg::ChatCompletions => Endpoint::ChatCompletions,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub base_url: String,
    #[arg(long)]
    pub model: String,
    #[arg(long, value_enum, default_value_t = EndpointArg::Completions)]
    pub endpoint: EndpointArg,
    #[arg(long, default_value_t = 640)]
    pub num_requests: usize,
    #[arg(long, default_value_t = 320)]
    pub concurrency: usize,
    #[arg(long, default_value_t = 0.0)]
    pub request_rate: f64,
    #[arg(long, default_value_t = 256)]
    pub input_tokens: usize,
    #[arg(long, default_value_t = 256)]
    pub output_tokens: usize,
    #[arg(long, default_value_t = true)]
    pub stream: bool,
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    #[arg(long, default_value_t = 500)]
    pub top_k: i32,
    #[arg(long, default_value_t = 0.3)]
    pub top_p: f32,
    #[arg(long, default_value_t = 0.5)]
    pub presence_penalty: f32,
    #[arg(long, default_value_t = 0.5)]
    pub repetition_penalty: f32,
    #[arg(long, default_value_t = 0.996)]
    pub penalty_decay: f32,
    #[arg(long, default_value_t = 0)]
    pub timeout_secs: u64,
    #[arg(long)]
    pub api_key: Option<String>,
    #[arg(long)]
    pub output_json: Option<PathBuf>,
    #[arg(long)]
    pub report_dir: Option<PathBuf>,
}

impl From<ServeArgs> for ServeConfig {
    fn from(args: ServeArgs) -> Self {
        Self {
            base_url: args.base_url,
            model: args.model,
            endpoint: args.endpoint.into(),
            num_requests: args.num_requests,
            concurrency: args.concurrency,
            request_rate: sanitize_request_rate(args.request_rate),
            input_tokens: args.input_tokens,
            output_tokens: args.output_tokens,
            stream: args.stream,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            presence_penalty: args.presence_penalty,
            repetition_penalty: args.repetition_penalty,
            penalty_decay: args.penalty_decay,
            timeout_secs: args.timeout_secs,
            api_key: args.api_key,
            metadata: BTreeMap::new(),
        }
    }
}

pub fn sanitize_request_rate(request_rate: f64) -> f64 {
    if request_rate.is_finite() && request_rate > 0.0 {
        request_rate
    } else {
        0.0
    }
}

pub fn default_output_path(name: &str) -> PathBuf {
    paths::bench_output_path(name)
}

pub fn normalized_args() -> Vec<OsString> {
    let mut args = std::env::args_os().collect::<Vec<_>>();
    if args.last() == Some(&OsString::from("--bench")) {
        args.pop();
    }
    args
}

pub fn build_runtime() -> Result<tokio::runtime::Runtime, std::io::Error> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
}
