use rwkv_config::raw::eval::SpaceDbConfig;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgSslMode};

use super::writes::recover_running_tasks;
use crate::db::Db;

pub async fn connect(cfg: &SpaceDbConfig, max_connections: u32) -> Result<Db, String> {
    let options = build_connect_options(cfg)?;
    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .connect_with(options)
        .await
        .map_err(|err| format!("connect postgres failed: {err}"))?;

    let db = Db { pool };
    let stats = recover_running_tasks(&db).await?;
    println!(
        "startup recovery: marked {} running tasks as Failed, {} running completions as Failed",
        stats.failed_task_count, stats.failed_completion_count
    );

    Ok(db)
}

fn build_connect_options(cfg: &SpaceDbConfig) -> Result<PgConnectOptions, String> {
    let mut options = PgConnectOptions::new()
        .host(&cfg.host)
        .port(parse_port(cfg)?)
        .username(&cfg.username)
        .database(&cfg.database_name)
        .ssl_mode(parse_ssl_mode(cfg.sslmode.as_deref())?);

    if !cfg.password.is_empty() {
        options = options.password(&cfg.password);
    }

    Ok(options)
}

fn parse_port(cfg: &SpaceDbConfig) -> Result<u16, String> {
    if cfg.port.trim().is_empty() {
        return Ok(5432);
    }

    cfg.port
        .trim()
        .parse::<u16>()
        .map_err(|err| format!("invalid postgres port `{}`: {err}", cfg.port))
}

fn parse_ssl_mode(value: Option<&str>) -> Result<PgSslMode, String> {
    match value
        .unwrap_or("prefer")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "disable" => Ok(PgSslMode::Disable),
        "prefer" => Ok(PgSslMode::Prefer),
        "require" => Ok(PgSslMode::Require),
        "verify-ca" => Ok(PgSslMode::VerifyCa),
        "verify-full" => Ok(PgSslMode::VerifyFull),
        other => Err(format!("unsupported postgres sslmode `{other}`")),
    }
}
