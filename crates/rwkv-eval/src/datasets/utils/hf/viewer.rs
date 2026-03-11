use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

const PARQUET_ENDPOINT: &str = "https://datasets-server.huggingface.co/parquet";
const ROWS_ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";

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
    let body = reqwest::Client::new()
        .get(PARQUET_ENDPOINT)
        .query(&[("dataset", dataset)])
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    serde_json::from_str::<ParquetResponse>(&body)
        .unwrap()
        .parquet_files
}

pub async fn get_split_row_count(dataset: &str, config: &str, split: &str) -> usize {
    let body = reqwest::Client::new()
        .get(ROWS_ENDPOINT)
        .query(&[
            ("dataset", dataset),
            ("config", config),
            ("split", split),
            ("offset", "0"),
            ("length", "1"),
        ])
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    serde_json::from_str::<Value>(&body).unwrap()["num_rows_total"]
        .as_u64()
        .unwrap() as usize
}
