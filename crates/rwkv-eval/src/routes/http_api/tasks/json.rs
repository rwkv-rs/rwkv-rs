use std::collections::BTreeMap;

use sonic_rs::{Value, from_str, prelude::*};

use crate::{
    db::TaskRecord,
    dtos::{ApiCotMode, SamplingSummary, ScoreSummary, TaskProgress},
    routes::http_api::error::ApiError,
};

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
    let mut metrics = parse_json_value(raw_metrics, "metrics")?;
    sanitize_score_metrics(&mut metrics);

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

pub(crate) fn build_task_progress(record: &TaskRecord, sampling: &SamplingSummary) -> TaskProgress {
    let planned_attempts = sampling
        .raw
        .get(&"planned_attempts")
        .and_then(Value::as_u64)
        .or_else(|| compute_planned_attempts(record, sampling))
        .unwrap_or_default();
    let completed_attempts = record.attempts_total.max(0) as u64;
    let percent = if planned_attempts == 0 {
        0.0
    } else {
        ((completed_attempts.min(planned_attempts)) as f64 / planned_attempts as f64)
            .clamp(0.0, 1.0)
    };

    TaskProgress {
        planned_attempts,
        completed_attempts,
        passed_attempts: record.attempts_passed.max(0) as u64,
        failed_attempts: record.attempts_failed.max(0) as u64,
        attempts_with_checker: record.attempts_with_checker.max(0) as u64,
        attempts_missing_checker: record.attempts_missing_checker.max(0) as u64,
        attempts_needing_human_review: record.attempts_needing_human_review.max(0) as u64,
        percent,
    }
}

fn sanitize_score_metrics(metrics: &mut Value) {
    if let Some(object) = metrics.as_object_mut() {
        object.remove(&"raw_success_counts");
    }
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

fn compute_planned_attempts(record: &TaskRecord, sampling: &SamplingSummary) -> Option<u64> {
    let total_len = u64::try_from(record.num_samples).ok()?;
    let avg_k = sampling.avg_k?;
    if !avg_k.is_finite() || avg_k <= 0.0 {
        return None;
    }

    let max_pass_k = sampling
        .raw
        .get(&"max_pass_k")
        .and_then(Value::as_u64)
        .or_else(|| sampling.pass_ks.iter().copied().max().map(u64::from))
        .unwrap_or(1);

    if avg_k < 1.0 {
        let sample_size = (((total_len as f64) * avg_k).round() as u64).clamp(1, total_len);
        return Some(sample_size.saturating_mul(max_pass_k));
    }

    let rounded = avg_k.round();
    if (avg_k - rounded).abs() > f64::EPSILON {
        return None;
    }
    let repeat_count = rounded as u64;
    Some(
        total_len
            .saturating_mul(repeat_count)
            .saturating_mul(max_pass_k),
    )
}
