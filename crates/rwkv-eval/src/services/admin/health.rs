use std::collections::{BTreeMap, BTreeSet};

use reqwest::Client;
use rwkv_config::raw::eval::RawEvalConfig;
use sonic_rs::Value;

use crate::services::{ServiceError, ServiceResult};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthTarget {
    pub base_url: String,
    pub roles: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct HealthTargetResult {
    pub base_url: String,
    pub roles: Vec<String>,
    pub status: String,
    pub error: Option<String>,
    pub health: Option<Value>,
}

pub async fn fetch_health_targets(
    client: &Client,
    cfg: &RawEvalConfig,
) -> ServiceResult<Vec<HealthTargetResult>> {
    let targets = collect_health_targets(cfg);
    let mut results = Vec::with_capacity(targets.len());

    for target in targets {
        let url = normalize_base_url(&target.base_url);
        let endpoint = format!("{}/health", url.trim_end_matches('/'));
        match client.get(&endpoint).send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    results.push(HealthTargetResult {
                        base_url: target.base_url,
                        roles: target.roles,
                        status: "unreachable".to_string(),
                        error: Some(format!("GET {endpoint} returned HTTP {status}")),
                        health: None,
                    });
                    continue;
                }

                let health = response.json::<Value>().await.map_err(|err| {
                    ServiceError::internal(format!("decode {endpoint} failed: {err}"))
                })?;
                results.push(HealthTargetResult {
                    base_url: target.base_url,
                    roles: target.roles,
                    status: "ok".to_string(),
                    error: None,
                    health: Some(health),
                });
            }
            Err(err) => results.push(HealthTargetResult {
                base_url: target.base_url,
                roles: target.roles,
                status: "unreachable".to_string(),
                error: Some(format!("GET {endpoint} failed: {err}")),
                health: None,
            }),
        }
    }

    Ok(results)
}

pub fn collect_health_targets(cfg: &RawEvalConfig) -> Vec<HealthTarget> {
    let mut dedup = BTreeMap::<String, BTreeSet<String>>::new();

    for model in &cfg.models {
        dedup
            .entry(model.base_url.clone())
            .or_default()
            .insert(format!("model:{}", model.model));
    }
    dedup
        .entry(cfg.llm_judger.base_url.clone())
        .or_default()
        .insert(format!("judger:{}", cfg.llm_judger.model));
    dedup
        .entry(cfg.llm_checker.base_url.clone())
        .or_default()
        .insert(format!("checker:{}", cfg.llm_checker.model));

    dedup
        .into_iter()
        .map(|(base_url, roles)| HealthTarget {
            base_url,
            roles: roles.into_iter().collect(),
        })
        .collect()
}

pub fn normalize_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

#[cfg(test)]
mod tests {
    use rwkv_config::raw::eval::{ExtApiConfig, IntApiConfig, RawEvalConfig, SpaceDbConfig};

    use super::{collect_health_targets, normalize_base_url};

    fn sample_cfg() -> RawEvalConfig {
        RawEvalConfig {
            experiment_name: "demo".to_string(),
            experiment_desc: "demo".to_string(),
            admin_api_key: None,
            run_mode: Some("new".to_string()),
            skip_checker: Some(false),
            judger_concurrency: Some(1),
            checker_concurrency: Some(1),
            db_pool_max_connections: Some(1),
            model_arch_versions: vec!["rwkv7".to_string()],
            model_data_versions: vec!["g1".to_string()],
            model_num_params: vec!["1.5b".to_string()],
            benchmark_field: vec!["Knowledge".to_string()],
            extra_benchmark_name: vec![],
            upload_to_space: Some(true),
            git_hash: "abc".to_string(),
            models: vec![
                IntApiConfig {
                    model_arch_version: "rwkv7".to_string(),
                    model_data_version: "g1".to_string(),
                    model_num_params: "1.5b".to_string(),
                    base_url: "127.0.0.1:8080".to_string(),
                    api_key: "secret".to_string(),
                    model: "demo-a".to_string(),
                    max_batch_size: Some(1),
                },
                IntApiConfig {
                    model_arch_version: "rwkv7".to_string(),
                    model_data_version: "g1".to_string(),
                    model_num_params: "3b".to_string(),
                    base_url: "127.0.0.1:8080".to_string(),
                    api_key: "secret".to_string(),
                    model: "demo-b".to_string(),
                    max_batch_size: Some(1),
                },
            ],
            llm_judger: ExtApiConfig {
                base_url: "judge:8080".to_string(),
                api_key: "secret".to_string(),
                model: "judge".to_string(),
            },
            llm_checker: ExtApiConfig {
                base_url: "judge:8080".to_string(),
                api_key: "secret".to_string(),
                model: "checker".to_string(),
            },
            space_db: SpaceDbConfig::default(),
        }
    }

    #[test]
    fn collects_unique_base_urls() {
        let targets = collect_health_targets(&sample_cfg());
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().any(|target| target.roles.len() == 2));
    }

    #[test]
    fn normalizes_scheme() {
        assert_eq!(
            normalize_base_url("127.0.0.1:8080"),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            normalize_base_url("https://example.com"),
            "https://example.com"
        );
    }
}
