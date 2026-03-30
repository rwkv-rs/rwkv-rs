use std::{collections::BTreeMap, sync::OnceLock};

use crate::{
    cores::datasets::{ALL_BENCHMARKS, BenchmarkInfo},
    db::{BenchmarkRecord, Db, ModelRecord, list_benchmarks, list_models},
    dtos::{ApiCotMode, BenchmarkField},
    services::{ServiceError, ServiceResult},
};

#[derive(Clone)]
pub struct BenchmarkCatalogEntry {
    pub display_name: &'static str,
    pub field: BenchmarkField,
    pub supported_cot_modes: Vec<ApiCotMode>,
    pub supported_n_shots: Vec<u8>,
    pub supported_avg_ks: Vec<f32>,
    pub supported_pass_ks: Vec<u8>,
}

pub async fn models(db: &Db) -> ServiceResult<Vec<ModelRecord>> {
    list_models(db).await.map_err(ServiceError::internal)
}

pub async fn benchmarks(db: &Db) -> ServiceResult<Vec<BenchmarkRecord>> {
    list_benchmarks(db).await.map_err(ServiceError::internal)
}

pub async fn meta(db: &Db) -> ServiceResult<(Vec<ModelRecord>, Vec<BenchmarkRecord>)> {
    Ok((models(db).await?, benchmarks(db).await?))
}

pub fn benchmark_names_for_field(field: BenchmarkField) -> Vec<String> {
    benchmark_catalog()
        .iter()
        .filter(|(_, entry)| entry.field == field)
        .map(|(name, _)| (*name).to_string())
        .collect()
}

pub fn catalog_for_benchmark(name: &str) -> Option<&'static BenchmarkCatalogEntry> {
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
