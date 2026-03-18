use std::collections::BTreeMap;

use sonic_rs::{Value, from_str, prelude::*};

use super::error::ApiError;
use super::schema::{ApiCotMode, SamplingSummary, ScoreSummary};
use crate::db::TaskRecord;

pub(crate) fn parse_sampling_summary(raw: &str) -> Result<SamplingSummary, ApiError> {
    let value = parse_json_value(raw, "sampling_config")?;
    let cot_mode = value
        .get(&"cot_mode")
        .and_then(Value::as_str)
        .and_then(ApiCotMode::parse);

    Ok(SamplingSummary {
        cot_mode,
        n_shot: value
            .get(&"n_shot")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        avg_k: value.get(&"avg_k").and_then(Value::as_f64),
        pass_ks: parse_pass_ks(&value),
        judger_model_name: parse_string_field(&value, "judger_model_name"),
        checker_model_name: parse_string_field(&value, "checker_model_name"),
        raw: value,
    })
}

pub(crate) fn parse_score_summary(record: &TaskRecord) -> Result<Option<ScoreSummary>, ApiError> {
    let Some(score_id) = record.score_id else {
        return Ok(None);
    };
    let raw_metrics = record
        .metrics_json
        .as_deref()
        .ok_or_else(|| ApiError::internal("score row missing metrics_json"))?;
    let metrics = parse_json_value(raw_metrics, "metrics")?;

    Ok(Some(ScoreSummary {
        score_id,
        created_at: record
            .score_created_at
            .clone()
            .ok_or_else(|| ApiError::internal("score row missing created_at"))?,
        cot_mode: record.score_cot_mode.as_deref().and_then(ApiCotMode::parse),
        passed: parse_u64_field(&metrics, "passed"),
        total: parse_u64_field(&metrics, "total"),
        sample_size: parse_u64_field(&metrics, "sample_size"),
        avg_repeat_count: parse_u64_field(&metrics, "avg_repeat_count"),
        max_pass_k: parse_u64_field(&metrics, "max_pass_k"),
        pass_at_k: parse_pass_at_k(&metrics),
        metrics,
    }))
}

fn parse_json_value(raw: &str, field_name: &str) -> Result<Value, ApiError> {
    from_str(raw).map_err(|err| {
        ApiError::internal(format!(
            "failed to parse {field_name} json from database: {err}"
        ))
    })
}

fn parse_u64_field(value: &Value, key: &str) -> Option<u64> {
    value.get(&key).and_then(Value::as_u64)
}

fn parse_string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(&key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn parse_pass_ks(value: &Value) -> Vec<u8> {
    value
        .get(&"pass_ks")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|item| u8::try_from(item).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_pass_at_k(value: &Value) -> BTreeMap<String, f64> {
    value
        .get(&"pass_at_k")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(key, value)| value.as_f64().map(|score| (key.to_string(), score)))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{parse_sampling_summary, parse_score_summary};
    use crate::db::{TaskRecord, TaskStatus};
    use crate::http_api::schema::ApiCotMode;

    #[test]
    fn parses_sampling_summary_with_sonic() {
        let summary = parse_sampling_summary(
            r#"{
                "cot_mode":"NoCoT",
                "n_shot":5,
                "avg_k":2.0,
                "pass_ks":[1,3],
                "judger_model_name":"judge-a",
                "checker_model_name":"checker-a"
            }"#,
        )
        .unwrap();

        assert_eq!(summary.cot_mode, Some(ApiCotMode::NoCot));
        assert_eq!(summary.n_shot, Some(5));
        assert_eq!(summary.pass_ks, vec![1, 3]);
        assert_eq!(summary.judger_model_name.as_deref(), Some("judge-a"));
    }

    #[test]
    fn parses_score_summary_with_sonic() {
        let record = TaskRecord {
            task_id: 1,
            config_path: None,
            evaluator: "rwkv-lm-eval".to_string(),
            is_param_search: false,
            is_tmp: false,
            task_created_at: "now".to_string(),
            task_status: TaskStatus::Completed,
            git_hash: "abc".to_string(),
            task_desc: None,
            sampling_config_json: "{}".to_string(),
            log_path: "/tmp/x.log".to_string(),
            model_id: 1,
            model_name: "model".to_string(),
            arch_version: "arch".to_string(),
            data_version: "data".to_string(),
            num_params: "7b".to_string(),
            benchmark_id: 1,
            benchmark_name: "bench".to_string(),
            benchmark_split: "test".to_string(),
            benchmark_url: None,
            benchmark_status: "Completed".to_string(),
            num_samples: 10,
            score_id: Some(9),
            score_created_at: Some("now".to_string()),
            score_cot_mode: Some("NoCoT".to_string()),
            metrics_json: Some(
                r#"{
                    "passed":6,
                    "total":10,
                    "sample_size":10,
                    "avg_repeat_count":1,
                    "max_pass_k":3,
                    "pass_at_k":{"pass@1":0.4,"pass@3":0.6}
                }"#
                .to_string(),
            ),
        };

        let summary = parse_score_summary(&record).unwrap().unwrap();
        assert_eq!(summary.cot_mode, Some(ApiCotMode::NoCot));
        assert_eq!(summary.passed, Some(6));
        assert_eq!(summary.pass_at_k.get("pass@3"), Some(&0.6));
    }
}
