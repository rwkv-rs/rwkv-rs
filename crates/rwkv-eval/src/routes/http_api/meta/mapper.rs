use crate::{
    db::{BenchmarkRecord, ModelRecord},
    dtos::{BenchmarkField, BenchmarkResource, ModelResource},
    services::meta::catalog_for_benchmark,
};

pub(crate) fn to_model_resource(model: &ModelRecord) -> ModelResource {
    ModelResource {
        model_id: model.model_id,
        model_name: model.model_name.clone(),
        arch_version: model.arch_version.clone(),
        data_version: model.data_version.clone(),
        num_params: model.num_params.clone(),
        model_version: model.model_version.clone(),
    }
}

pub(crate) fn to_benchmark_resource(benchmark: &BenchmarkRecord) -> BenchmarkResource {
    let catalog = catalog_for_benchmark(&benchmark.benchmark_name);

    BenchmarkResource {
        benchmark_id: benchmark.benchmark_id,
        benchmark_name: benchmark.benchmark_name.clone(),
        display_name: catalog
            .map(|entry| entry.display_name.to_string())
            .unwrap_or_else(|| benchmark.benchmark_name.clone()),
        field: catalog
            .map(|entry| entry.field)
            .unwrap_or(BenchmarkField::Unknown),
        benchmark_split: benchmark.benchmark_split.clone(),
        status: benchmark.status.clone(),
        num_samples: benchmark.num_samples,
        url: benchmark.url.clone(),
        supported_cot_modes: catalog
            .map(|entry| entry.supported_cot_modes.clone())
            .unwrap_or_default(),
        supported_n_shots: catalog
            .map(|entry| entry.supported_n_shots.clone())
            .unwrap_or_default(),
        supported_avg_ks: catalog
            .map(|entry| entry.supported_avg_ks.clone())
            .unwrap_or_default(),
        supported_pass_ks: catalog
            .map(|entry| entry.supported_pass_ks.clone())
            .unwrap_or_default(),
    }
}
