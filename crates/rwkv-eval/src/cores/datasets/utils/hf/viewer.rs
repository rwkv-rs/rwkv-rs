use std::{path::PathBuf, time::Duration};

use once_cell::sync::Lazy;
use serde::Deserialize;
use sonic_rs::{Value, prelude::*};
use tokio::{sync::Semaphore, time::sleep};

const PARQUET_ENDPOINT: &str = "https://datasets-server.huggingface.co/parquet";
const ROWS_ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";
const VIEWER_RETRY_ATTEMPTS: usize = 3;
const VIEWER_REQUEST_DELAY_MS: u64 = 200;
static VIEWER_REQUEST_SEMAPHORE: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(1));

#[derive(Debug, Clone, Deserialize)]
pub struct ParquetFile {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub url: String,
    pub filename: String,
    pub size: usize,
}

impl ParquetFile {
    pub fn relative_path(&self) -> PathBuf {
        PathBuf::from(&self.config)
            .join(&self.split)
            .join(&self.filename)
    }
}

#[derive(Debug, Deserialize)]
struct ParquetResponse {
    parquet_files: Vec<ParquetFile>,
}

pub async fn get_parquet_files(dataset: &str) -> Vec<ParquetFile> {
    let body = fetch_viewer_body(
        PARQUET_ENDPOINT,
        &[("dataset", dataset)],
        &format!("fetch parquet files dataset={dataset}"),
    )
    .await;

    sonic_rs::from_str::<ParquetResponse>(&body)
        .unwrap_or_else(|err| panic!("failed to parse parquet response dataset={dataset}: {err}"))
        .parquet_files
}

pub async fn get_split_row_count(dataset: &str, config: &str, split: &str) -> usize {
    let body = fetch_viewer_body(
        ROWS_ENDPOINT,
        &[
            ("dataset", dataset),
            ("config", config),
            ("split", split),
            ("offset", "0"),
            ("length", "1"),
        ],
        &format!("fetch split row count dataset={dataset} config={config} split={split}"),
    )
    .await;

    sonic_rs::from_str::<Value>(&body).unwrap_or_else(|err| {
        panic!(
            "failed to parse split row count response dataset={dataset} config={config} split={split}: {err}"
        )
    })["num_rows_total"]
        .as_u64()
        .unwrap_or_else(|| {
            panic!(
                "split row count response missing num_rows_total dataset={dataset} config={config} split={split}"
            )
        }) as usize
}

async fn fetch_viewer_body(endpoint: &str, query: &[(&str, &str)], operation_name: &str) -> String {
    let _permit = VIEWER_REQUEST_SEMAPHORE.acquire().await.unwrap();
    let client = reqwest::Client::new();
    let mut last_error = String::new();

    for attempt in 1..=VIEWER_RETRY_ATTEMPTS {
        sleep(Duration::from_millis(VIEWER_REQUEST_DELAY_MS)).await;
        match client.get(endpoint).query(query).send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    last_error = format!("unexpected HTTP status: {status}");
                } else {
                    return response.text().await.unwrap_or_else(|err| {
                        panic!("{operation_name} failed to read response body: {err}")
                    });
                }
            }
            Err(err) => {
                last_error = err.to_string();
            }
        }

        if attempt < VIEWER_RETRY_ATTEMPTS {
            let backoff_secs = 1_u64 << (attempt - 1);
            eprintln!(
                "[network retry {}/{}] {} failed: {}. Retrying in {}s...",
                attempt, VIEWER_RETRY_ATTEMPTS, operation_name, last_error, backoff_secs
            );
            sleep(Duration::from_secs(backoff_secs)).await;
        }
    }

    panic!(
        "network operation failed after {} attempts: {}. last error: {}",
        VIEWER_RETRY_ATTEMPTS, operation_name, last_error
    );
}
