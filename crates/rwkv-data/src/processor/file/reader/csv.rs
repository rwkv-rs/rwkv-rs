use std::{
    borrow::Cow,
    future::Future,
    marker::PhantomData,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use csv_async::{AsyncReaderBuilder, Trim};
use futures::StreamExt;
use serde::Deserialize;
use tokio::{
    fs::File,
    sync::mpsc::{Receiver, channel},
};

use crate::processor::file::{DataItem, Reader, find_common_prefix, generate_relative_name};

pub struct CsvReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    paths: Vec<PathBuf>,
    common_prefix: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> CsvReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    pub fn new(paths: Vec<PathBuf>, converter: F) -> Self {
        let common_prefix = find_common_prefix(&paths);

        Self {
            paths,
            common_prefix,
            converter: Arc::new(converter),
            phantom: PhantomData,
        }
    }

    fn get_delimiter(path: &Path) -> u8 {
        match path.extension().and_then(|s| s.to_str()) {
            Some("tsv") => b'\t',
            _ => b',',
        }
    }
}

impl<T, F> Reader for CsvReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>> {
        let paths = self.paths.clone();

        let common_prefix = self.common_prefix.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let (tx, rx) = channel(40960);

            tokio::spawn(async move {
                const BATCH_SIZE: usize = 4096;

                for path in paths {
                    let relative_name = generate_relative_name(&path, &common_prefix);

                    tx.send(DataItem::FileStart(relative_name)).await.unwrap();

                    let delimiter = Self::get_delimiter(&path);

                    let file = File::open(&path).await.unwrap();

                    let mut reader = AsyncReaderBuilder::new()
                        .delimiter(delimiter)
                        .trim(Trim::All)
                        .create_reader(file);

                    let mut batch: Vec<Cow<'static, str>> = Vec::with_capacity(BATCH_SIZE);

                    while let Some(record) = reader.records().next().await {
                        let record = record.unwrap();

                        let parsed: T = record.deserialize(None).unwrap();

                        let converted = converter(parsed);

                        batch.push(converted);

                        if batch.len() >= BATCH_SIZE {
                            let to_send = std::mem::take(&mut batch);

                            if tx.send(DataItem::DataBatch(to_send)).await.is_err() {
                                return;
                            }

                            batch = Vec::with_capacity(BATCH_SIZE);
                        }
                    }

                    if !batch.is_empty() {
                        let to_send = std::mem::take(&mut batch);

                        if tx.send(DataItem::DataBatch(to_send)).await.is_err() {
                            return;
                        }
                    }
                }
            });

            rx
        })
    }
}
