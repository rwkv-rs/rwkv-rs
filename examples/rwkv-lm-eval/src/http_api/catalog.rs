use std::{collections::BTreeMap, sync::OnceLock};

use rwkv_eval::datasets::{ALL_BENCHMARKS, BenchmarkInfo};

use super::schema::{ApiCotMode, BenchmarkField};

#[derive(Clone)]
pub(crate) struct BenchmarkCatalogEntry {
    pub display_name: &'static str,
    pub field: BenchmarkField,
    pub supported_cot_modes: Vec<ApiCotMode>,
    pub supported_n_shots: Vec<u8>,
    pub supported_avg_ks: Vec<f32>,
    pub supported_pass_ks: Vec<u8>,
}

pub(crate) fn benchmark_names_for_field(field: BenchmarkField) -> Vec<String> {
    benchmark_catalog()
        .iter()
        .filter(|(_, entry)| entry.field == field)
        .map(|(name, _)| (*name).to_string())
        .collect()
}

pub(crate) fn catalog_for_benchmark(name: &str) -> Option<&'static BenchmarkCatalogEntry> {
    benchmark_catalog().get(name)
}

fn benchmark_catalog() -> &'static BTreeMap<&'static str, BenchmarkCatalogEntry> {
    static CATALOG: OnceLock<BTreeMap<&'static str, BenchmarkCatalogEntry>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        ALL_BENCHMARKS
            .iter()
            .map(|info| (info.name.0, catalog_entry(info)))
            .collect()
    })
}

fn catalog_entry(info: &BenchmarkInfo) -> BenchmarkCatalogEntry {
    BenchmarkCatalogEntry {
        display_name: info.display_name,
        field: BenchmarkField::from_rwkv(info.field),
        supported_cot_modes: info
            .cot_mode
            .iter()
            .copied()
            .map(ApiCotMode::from_eval_mode)
            .collect(),
        supported_n_shots: info.n_shots.to_vec(),
        supported_avg_ks: info.avg_ks.to_vec(),
        supported_pass_ks: info.pass_ks.to_vec(),
    }
}
