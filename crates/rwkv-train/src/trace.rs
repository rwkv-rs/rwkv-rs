pub use rwkv_bench::trace::{TraceConfig, TraceLevel, TraceMode};

pub fn init_tracing(service: &'static str) -> Result<TraceMode, String> {
    init_tracing_with(service, TraceConfig::default())
}

pub fn init_tracing_with(service: &'static str, cfg: TraceConfig) -> Result<TraceMode, String> {
    rwkv_bench::trace::init_tracing(service, cfg).map_err(|err| err.to_string())
}
