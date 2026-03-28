use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use flate2::read::GzDecoder;
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
        .map(|line| sonic_rs::from_str(&line).unwrap())
        .collect()
}

pub fn read_gzip_jsonl_items<T, P>(path: P) -> Vec<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref()).unwrap();
    let reader = BufReader::new(GzDecoder::new(file));

    reader
        .lines()
        .map(|line| line.unwrap())
        .filter(|line| !line.trim().is_empty())
        .map(|line| sonic_rs::from_str(&line).unwrap())
        .collect()
}
