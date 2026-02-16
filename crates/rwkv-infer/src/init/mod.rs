use std::path::Path;

use clia_tracing_config::WorkerGuard;
use rwkv_config::load_toml;
use rwkv_config::raw::infer::RawInferConfig;
use rwkv_config::validated::infer::FinalInferConfigBuilder;

pub fn init_cfg<P: AsRef<Path>>(infer_cfg_path: P) -> FinalInferConfigBuilder {
    let mut raw: RawInferConfig = load_toml(infer_cfg_path);
    raw.fill_default();

    let builder = FinalInferConfigBuilder::load_from_raw(raw);
    builder.check();
    builder
}

pub fn init_log(level: &str) -> WorkerGuard {
    clia_tracing_config::build()
        .filter_level(level)
        .with_ansi(true)
        .to_stdout(true)
        .init()
}

