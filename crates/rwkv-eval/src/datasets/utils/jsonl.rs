use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::de::DeserializeOwned;

pub fn read_jsonl_items<T, P>(path: P) -> Vec<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref()).unwrap();
    let reader = BufReader::new(file);

    reader
        .lines()
        .map(|line| line.unwrap())
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(&line).unwrap())
        .collect()
}
