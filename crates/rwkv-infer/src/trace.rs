pub use rwkv_bench::trace::{TraceConfig, TraceLevel, TraceMode};

pub fn init_tracing(service: &'static str) -> crate::Result<TraceMode> {
    init_tracing_with(service, TraceConfig::default())
}

pub fn init_tracing_with(service: &'static str, cfg: TraceConfig) -> crate::Result<TraceMode> {
    rwkv_bench::trace::init_tracing(service, cfg)
        .map_err(|err| crate::Error::internal(format!("failed to initialize tracing: {err}")))
}
