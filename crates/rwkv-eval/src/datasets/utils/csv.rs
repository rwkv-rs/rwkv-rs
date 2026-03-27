use std::{fs::File, path::Path};

use serde::de::DeserializeOwned;

pub fn read_csv_items<T, P>(path: P) -> Vec<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref()).unwrap();
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(file);

    reader.deserialize().map(|row| row.unwrap()).collect()
}
