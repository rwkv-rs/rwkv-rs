use sqlx::{Row, query};

use crate::db::{BenchmarkRecord, Db, ModelRecord};

pub async fn list_models(db: &Db) -> Result<Vec<ModelRecord>, String> {
    let rows = query(
        r#"
        SELECT
            model_id,
            model_name,
            arch_version,
            data_version,
            num_params,
            model_version
        FROM view_model_version
        ORDER BY arch_version, num_params, data_version, model_name
        "#,
    )
    .fetch_all(&db.pool)
    .await
    .map_err(|err| format!("list models failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            Ok(ModelRecord {
                model_id: row
                    .try_get("model_id")
                    .map_err(|err| format!("decode model_id failed: {err}"))?,
                model_name: row
                    .try_get("model_name")
                    .map_err(|err| format!("decode model_name failed: {err}"))?,
                arch_version: row
                    .try_get("arch_version")
                    .map_err(|err| format!("decode arch_version failed: {err}"))?,
                data_version: row
                    .try_get("data_version")
                    .map_err(|err| format!("decode data_version failed: {err}"))?,
                num_params: row
                    .try_get("num_params")
                    .map_err(|err| format!("decode num_params failed: {err}"))?,
                model_version: row
                    .try_get("model_version")
                    .map_err(|err| format!("decode model_version failed: {err}"))?,
            })
        })
        .collect()
}

pub async fn list_benchmarks(db: &Db) -> Result<Vec<BenchmarkRecord>, String> {
    let rows = query(
        r#"
        SELECT
            benchmark_id,
            benchmark_name,
            benchmark_split,
            url,
            status,
            num_samples
        FROM benchmark
        ORDER BY benchmark_name, benchmark_split
        "#,
    )
    .fetch_all(&db.pool)
    .await
    .map_err(|err| format!("list benchmarks failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            Ok(BenchmarkRecord {
                benchmark_id: row
                    .try_get("benchmark_id")
                    .map_err(|err| format!("decode benchmark_id failed: {err}"))?,
                benchmark_name: row
                    .try_get("benchmark_name")
                    .map_err(|err| format!("decode benchmark_name failed: {err}"))?,
                benchmark_split: row
                    .try_get("benchmark_split")
                    .map_err(|err| format!("decode benchmark_split failed: {err}"))?,
                url: row
                    .try_get("url")
                    .map_err(|err| format!("decode benchmark url failed: {err}"))?,
                status: row
                    .try_get("status")
                    .map_err(|err| format!("decode benchmark status failed: {err}"))?,
                num_samples: row
                    .try_get("num_samples")
                    .map_err(|err| format!("decode benchmark num_samples failed: {err}"))?,
            })
        })
        .collect()
}
