use serde_json::Value;

const ROWS_ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";

pub(crate) async fn get_split_row_count(dataset: &str, config: &str, split: &str) -> usize {
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

    parse_num_rows_total(&body)
}

fn parse_num_rows_total(body: &str) -> usize {
    serde_json::from_str::<Value>(body).unwrap()["num_rows_total"]
        .as_u64()
        .unwrap() as usize
}

#[cfg(test)]
mod tests {
    use super::parse_num_rows_total;

    #[test]
    fn parse_num_rows_total_reads_expected_field() {
        assert_eq!(parse_num_rows_total(r#"{"num_rows_total": 285}"#), 285);
    }
}
