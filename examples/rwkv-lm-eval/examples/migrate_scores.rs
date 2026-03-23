use std::collections::HashMap;

use clap::Parser;
use sonic_rs::{Value, from_str, prelude::*};
use sqlx::{Postgres, Row, Transaction, postgres::PgPoolOptions, query};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    source_db_url: String,
    #[arg(long)]
    target_db_url: String,
    #[arg(long, default_value_t = false)]
    apply: bool,
    #[arg(long, default_value_t = false)]
    overwrite_existing: bool,
    #[arg(long, default_value_t = false)]
    prefer_latest_match: bool,
    #[arg(long, default_value_t = false)]
    ignore_config_path: bool,
    #[arg(long, default_value_t = false)]
    ignore_git_hash: bool,
    #[arg(long, default_value_t = false)]
    keep_legacy_metrics: bool,
    #[arg(long)]
    limit: Option<i64>,
    #[arg(long, default_value_t = 10)]
    max_issue_examples: usize,
}

#[derive(Clone, Debug)]
struct SourceScoreRow {
    score_id: i32,
    task_id: i32,
    config_path: Option<String>,
    evaluator: String,
    git_hash: String,
    sampling_config_json: String,
    model_name: String,
    arch_version: String,
    data_version: String,
    num_params: String,
    benchmark_name: String,
    benchmark_split: String,
    cot_mode: String,
    metrics_json: String,
    created_at: String,
}

#[derive(Clone, Debug)]
struct TargetTaskRow {
    task_id: i32,
    existing_score_id: Option<i32>,
    identity: TaskIdentityKey,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TaskIdentityKey {
    config_path: Option<String>,
    evaluator: String,
    git_hash: Option<String>,
    sampling_config_json: String,
    model_name: String,
    arch_version: String,
    data_version: String,
    num_params: String,
    benchmark_name: String,
    benchmark_split: String,
}

#[derive(Default)]
struct MigrationStats {
    source_rows: usize,
    matched_rows: usize,
    missing_target_rows: usize,
    ambiguous_target_rows: usize,
    existing_target_rows: usize,
    would_insert_rows: usize,
    would_update_rows: usize,
    inserted_rows: usize,
    updated_rows: usize,
}

struct Issue {
    kind: &'static str,
    score_id: i32,
    task_id: i32,
    detail: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!(
        "legacy score migration starting (mode: {})",
        if args.apply { "apply" } else { "dry-run" }
    );
    println!(
        "match options: ignore_config_path={} ignore_git_hash={} prefer_latest_match={} overwrite_existing={} keep_legacy_metrics={}",
        args.ignore_config_path,
        args.ignore_git_hash,
        args.prefer_latest_match,
        args.overwrite_existing,
        args.keep_legacy_metrics,
    );

    let source_pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&args.source_db_url)
        .await
        .unwrap_or_else(|err| panic!("failed to connect source postgres: {err}"));
    let target_pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&args.target_db_url)
        .await
        .unwrap_or_else(|err| panic!("failed to connect target postgres: {err}"));

    let source_rows = load_source_scores(&source_pool, args.limit)
        .await
        .unwrap_or_else(|err| panic!("failed to load source scores: {err}"));
    let target_rows = load_target_tasks(&target_pool)
        .await
        .unwrap_or_else(|err| panic!("failed to load target tasks: {err}"));
    let target_index = build_target_index(&args, &target_rows);

    println!(
        "loaded {} source score rows and {} target task rows",
        source_rows.len(),
        target_rows.len()
    );

    let mut stats = MigrationStats {
        source_rows: source_rows.len(),
        ..MigrationStats::default()
    };
    let mut issues = Vec::new();
    let mut tx = if args.apply {
        Some(
            target_pool
                .begin()
                .await
                .unwrap_or_else(|err| panic!("failed to start target transaction: {err}")),
        )
    } else {
        None
    };

    for source in &source_rows {
        let key = build_identity_key(
            &args,
            source.config_path.as_deref(),
            &source.evaluator,
            &source.git_hash,
            &source.sampling_config_json,
            &source.model_name,
            &source.arch_version,
            &source.data_version,
            &source.num_params,
            &source.benchmark_name,
            &source.benchmark_split,
        );
        let Some(candidates) = target_index.get(&key) else {
            stats.missing_target_rows += 1;
            record_issue(
                &mut issues,
                args.max_issue_examples,
                "missing-target-task",
                source,
                format!(
                    "no target task matched evaluator={} model={} benchmark={} sampling_config={}",
                    source.evaluator,
                    source.model_name,
                    source.benchmark_name,
                    source.sampling_config_json
                ),
            );
            continue;
        };

        let target = if candidates.len() == 1 {
            &candidates[0]
        } else if args.prefer_latest_match {
            candidates
                .last()
                .unwrap_or_else(|| panic!("target candidates unexpectedly empty"))
        } else {
            stats.ambiguous_target_rows += 1;
            record_issue(
                &mut issues,
                args.max_issue_examples,
                "ambiguous-target-task",
                source,
                format!(
                    "matched multiple target task_ids={:?}",
                    candidates
                        .iter()
                        .map(|task| task.task_id)
                        .collect::<Vec<_>>()
                ),
            );
            continue;
        };

        stats.matched_rows += 1;

        if target.existing_score_id.is_some() && !args.overwrite_existing {
            stats.existing_target_rows += 1;
            record_issue(
                &mut issues,
                args.max_issue_examples,
                "existing-target-score",
                source,
                format!(
                    "target task_id={} already has score_id={}",
                    target.task_id,
                    target
                        .existing_score_id
                        .unwrap_or_else(|| panic!("existing score id unexpectedly missing"))
                ),
            );
            continue;
        }

        let normalized_metrics_json = normalize_metrics_json(
            &source.metrics_json,
            args.keep_legacy_metrics,
            source.score_id,
        )
        .unwrap_or_else(|err| panic!("{err}"));

        if target.existing_score_id.is_some() {
            stats.would_update_rows += 1;
        } else {
            stats.would_insert_rows += 1;
        }

        if let Some(tx) = tx.as_mut() {
            upsert_score(
                tx,
                target.task_id,
                &source.cot_mode,
                &normalized_metrics_json,
                &source.created_at,
                source.score_id,
            )
            .await
            .unwrap_or_else(|err| panic!("{err}"));

            if target.existing_score_id.is_some() {
                stats.updated_rows += 1;
            } else {
                stats.inserted_rows += 1;
            }
        }
    }

    if let Some(tx) = tx {
        tx.commit()
            .await
            .unwrap_or_else(|err| panic!("failed to commit target transaction: {err}"));
    }

    print_summary(&args, &stats, &issues);
}

async fn load_source_scores(
    pool: &sqlx::PgPool,
    limit: Option<i64>,
) -> Result<Vec<SourceScoreRow>, String> {
    let sql = r#"
        SELECT
            s.score_id,
            s.task_id,
            t.config_path,
            t.evaluator,
            t.git_hash,
            t.sampling_config::text AS sampling_config_json,
            m.model_name,
            m.arch_version,
            m.data_version,
            m.num_params,
            b.benchmark_name,
            b.benchmark_split,
            s.cot_mode,
            s.metrics::text AS metrics_json,
            s.created_at::text AS created_at
        FROM scores s
        JOIN task t ON t.task_id = s.task_id
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        ORDER BY s.score_id
    "#;
    let sql_with_limit = r#"
        SELECT
            s.score_id,
            s.task_id,
            t.config_path,
            t.evaluator,
            t.git_hash,
            t.sampling_config::text AS sampling_config_json,
            m.model_name,
            m.arch_version,
            m.data_version,
            m.num_params,
            b.benchmark_name,
            b.benchmark_split,
            s.cot_mode,
            s.metrics::text AS metrics_json,
            s.created_at::text AS created_at
        FROM scores s
        JOIN task t ON t.task_id = s.task_id
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        ORDER BY s.score_id
        LIMIT $1
    "#;

    let rows = if let Some(limit) = limit {
        query(sql_with_limit)
            .bind(limit)
            .fetch_all(pool)
            .await
            .map_err(|err| format!("query source scores with limit failed: {err}"))?
    } else {
        query(sql)
            .fetch_all(pool)
            .await
            .map_err(|err| format!("query source scores failed: {err}"))?
    };

    rows.into_iter()
        .map(|row| {
            Ok(SourceScoreRow {
                score_id: row
                    .try_get("score_id")
                    .map_err(|err| format!("decode source score_id failed: {err}"))?,
                task_id: row
                    .try_get("task_id")
                    .map_err(|err| format!("decode source task_id failed: {err}"))?,
                config_path: row
                    .try_get("config_path")
                    .map_err(|err| format!("decode source config_path failed: {err}"))?,
                evaluator: row
                    .try_get("evaluator")
                    .map_err(|err| format!("decode source evaluator failed: {err}"))?,
                git_hash: row
                    .try_get("git_hash")
                    .map_err(|err| format!("decode source git_hash failed: {err}"))?,
                sampling_config_json: row
                    .try_get("sampling_config_json")
                    .map_err(|err| format!("decode source sampling_config_json failed: {err}"))?,
                model_name: row
                    .try_get("model_name")
                    .map_err(|err| format!("decode source model_name failed: {err}"))?,
                arch_version: row
                    .try_get("arch_version")
                    .map_err(|err| format!("decode source arch_version failed: {err}"))?,
                data_version: row
                    .try_get("data_version")
                    .map_err(|err| format!("decode source data_version failed: {err}"))?,
                num_params: row
                    .try_get("num_params")
                    .map_err(|err| format!("decode source num_params failed: {err}"))?,
                benchmark_name: row
                    .try_get("benchmark_name")
                    .map_err(|err| format!("decode source benchmark_name failed: {err}"))?,
                benchmark_split: row
                    .try_get("benchmark_split")
                    .map_err(|err| format!("decode source benchmark_split failed: {err}"))?,
                cot_mode: row
                    .try_get("cot_mode")
                    .map_err(|err| format!("decode source cot_mode failed: {err}"))?,
                metrics_json: row
                    .try_get("metrics_json")
                    .map_err(|err| format!("decode source metrics_json failed: {err}"))?,
                created_at: row
                    .try_get("created_at")
                    .map_err(|err| format!("decode source created_at failed: {err}"))?,
            })
        })
        .collect()
}

async fn load_target_tasks(pool: &sqlx::PgPool) -> Result<Vec<TargetTaskRow>, String> {
    let rows = query(
        r#"
        SELECT
            t.task_id,
            t.config_path,
            t.evaluator,
            t.git_hash,
            t.sampling_config::text AS sampling_config_json,
            m.model_name,
            m.arch_version,
            m.data_version,
            m.num_params,
            b.benchmark_name,
            b.benchmark_split,
            s.score_id AS existing_score_id
        FROM task t
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        ORDER BY t.task_id
        "#,
    )
    .fetch_all(pool)
    .await
    .map_err(|err| format!("query target tasks failed: {err}"))?;

    rows.into_iter()
        .map(|row| {
            Ok(TargetTaskRow {
                task_id: row
                    .try_get("task_id")
                    .map_err(|err| format!("decode target task_id failed: {err}"))?,
                existing_score_id: row
                    .try_get("existing_score_id")
                    .map_err(|err| format!("decode target existing_score_id failed: {err}"))?,
                identity: TaskIdentityKey {
                    config_path: row
                        .try_get("config_path")
                        .map_err(|err| format!("decode target config_path failed: {err}"))?,
                    evaluator: row
                        .try_get("evaluator")
                        .map_err(|err| format!("decode target evaluator failed: {err}"))?,
                    git_hash: Some(
                        row.try_get("git_hash")
                            .map_err(|err| format!("decode target git_hash failed: {err}"))?,
                    ),
                    sampling_config_json: row.try_get("sampling_config_json").map_err(|err| {
                        format!("decode target sampling_config_json failed: {err}")
                    })?,
                    model_name: row
                        .try_get("model_name")
                        .map_err(|err| format!("decode target model_name failed: {err}"))?,
                    arch_version: row
                        .try_get("arch_version")
                        .map_err(|err| format!("decode target arch_version failed: {err}"))?,
                    data_version: row
                        .try_get("data_version")
                        .map_err(|err| format!("decode target data_version failed: {err}"))?,
                    num_params: row
                        .try_get("num_params")
                        .map_err(|err| format!("decode target num_params failed: {err}"))?,
                    benchmark_name: row
                        .try_get("benchmark_name")
                        .map_err(|err| format!("decode target benchmark_name failed: {err}"))?,
                    benchmark_split: row
                        .try_get("benchmark_split")
                        .map_err(|err| format!("decode target benchmark_split failed: {err}"))?,
                },
            })
        })
        .collect()
}

fn build_target_index(
    args: &Args,
    rows: &[TargetTaskRow],
) -> HashMap<TaskIdentityKey, Vec<TargetTaskRow>> {
    let mut index = HashMap::<TaskIdentityKey, Vec<TargetTaskRow>>::new();
    for row in rows {
        index
            .entry(normalize_identity_key(args, &row.identity))
            .or_default()
            .push(row.clone());
    }
    for tasks in index.values_mut() {
        tasks.sort_by_key(|task| task.task_id);
    }
    index
}

fn normalize_identity_key(args: &Args, identity: &TaskIdentityKey) -> TaskIdentityKey {
    TaskIdentityKey {
        config_path: if args.ignore_config_path {
            None
        } else {
            identity.config_path.clone()
        },
        evaluator: identity.evaluator.clone(),
        git_hash: if args.ignore_git_hash {
            None
        } else {
            identity.git_hash.clone()
        },
        sampling_config_json: identity.sampling_config_json.clone(),
        model_name: identity.model_name.clone(),
        arch_version: identity.arch_version.clone(),
        data_version: identity.data_version.clone(),
        num_params: identity.num_params.clone(),
        benchmark_name: identity.benchmark_name.clone(),
        benchmark_split: identity.benchmark_split.clone(),
    }
}

fn build_identity_key(
    args: &Args,
    config_path: Option<&str>,
    evaluator: &str,
    git_hash: &str,
    sampling_config_json: &str,
    model_name: &str,
    arch_version: &str,
    data_version: &str,
    num_params: &str,
    benchmark_name: &str,
    benchmark_split: &str,
) -> TaskIdentityKey {
    TaskIdentityKey {
        config_path: if args.ignore_config_path {
            None
        } else {
            config_path.map(ToOwned::to_owned)
        },
        evaluator: evaluator.to_string(),
        git_hash: if args.ignore_git_hash {
            None
        } else {
            Some(git_hash.to_string())
        },
        sampling_config_json: sampling_config_json.to_string(),
        model_name: model_name.to_string(),
        arch_version: arch_version.to_string(),
        data_version: data_version.to_string(),
        num_params: num_params.to_string(),
        benchmark_name: benchmark_name.to_string(),
        benchmark_split: benchmark_split.to_string(),
    }
}

fn normalize_metrics_json(
    raw_metrics_json: &str,
    keep_legacy_metrics: bool,
    source_score_id: i32,
) -> Result<String, String> {
    if keep_legacy_metrics {
        return Ok(raw_metrics_json.to_string());
    }

    let mut metrics: Value = from_str(raw_metrics_json).map_err(|err| {
        format!("parse source metrics json failed for score_id={source_score_id}: {err}")
    })?;
    if let Some(object) = metrics.as_object_mut() {
        object.remove(&"raw_success_counts");
    }

    sonic_rs::to_string(&metrics).map_err(|err| {
        format!("serialize normalized metrics json failed for score_id={source_score_id}: {err}")
    })
}

async fn upsert_score(
    tx: &mut Transaction<'_, Postgres>,
    task_id: i32,
    cot_mode: &str,
    metrics_json: &str,
    created_at: &str,
    source_score_id: i32,
) -> Result<(), String> {
    query(
        r#"
        INSERT INTO scores (task_id, cot_mode, metrics, created_at)
        VALUES ($1, $2, $3::jsonb, $4::timestamp)
        ON CONFLICT (task_id) DO UPDATE
        SET
            cot_mode = EXCLUDED.cot_mode,
            metrics = EXCLUDED.metrics,
            created_at = EXCLUDED.created_at
        "#,
    )
    .bind(task_id)
    .bind(cot_mode)
    .bind(metrics_json)
    .bind(created_at)
    .execute(&mut **tx)
    .await
    .map_err(|err| {
        format!(
            "upsert target score failed for source score_id={} target task_id={}: {}",
            source_score_id, task_id, err
        )
    })?;

    Ok(())
}

fn record_issue(
    issues: &mut Vec<Issue>,
    max_issue_examples: usize,
    kind: &'static str,
    source: &SourceScoreRow,
    detail: String,
) {
    if issues.len() >= max_issue_examples {
        return;
    }

    issues.push(Issue {
        kind,
        score_id: source.score_id,
        task_id: source.task_id,
        detail,
    });
}

fn print_summary(args: &Args, stats: &MigrationStats, issues: &[Issue]) {
    println!();
    println!("migration summary");
    println!("  mode: {}", if args.apply { "apply" } else { "dry-run" });
    println!("  source rows: {}", stats.source_rows);
    println!("  matched target tasks: {}", stats.matched_rows);
    println!("  missing target tasks: {}", stats.missing_target_rows);
    println!("  ambiguous target tasks: {}", stats.ambiguous_target_rows);
    println!(
        "  existing target scores skipped: {}",
        stats.existing_target_rows
    );
    println!("  would insert: {}", stats.would_insert_rows);
    println!("  would update: {}", stats.would_update_rows);
    if args.apply {
        println!("  inserted: {}", stats.inserted_rows);
        println!("  updated: {}", stats.updated_rows);
    } else {
        println!("  dry-run only; rerun with --apply to write changes");
    }

    if issues.is_empty() {
        return;
    }

    println!();
    println!("issue examples");
    for issue in issues {
        println!(
            "  [{}] source score_id={} task_id={}: {}",
            issue.kind, issue.score_id, issue.task_id, issue.detail
        );
    }
}
