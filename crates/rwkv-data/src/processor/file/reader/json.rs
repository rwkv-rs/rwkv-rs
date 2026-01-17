use std::{borrow::Cow, future::Future, marker::PhantomData, path::PathBuf, pin::Pin, sync::Arc};

use sonic_rs::{Deserialize, from_str};
use tokio::{
    fs::File,
    io::{AsyncBufReadExt, BufReader},
    sync::{mpsc, mpsc::Receiver},
};

use crate::processor::file::{DataItem, Reader, find_common_prefix, generate_relative_name};

pub struct JsonReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    paths: Vec<PathBuf>,
    common_prefix: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> JsonReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    pub fn new(paths: Vec<PathBuf>, converter: F) -> Self {
        let common_prefix = find_common_prefix(&paths);

        Self {
            paths: paths.iter().map(|p| p.to_path_buf()).collect(),
            common_prefix,
            converter: Arc::new(converter),
            phantom: PhantomData,
        }
    }
}

impl<T, F> Reader for JsonReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>> {
        let paths = self.paths.clone();

        let common_prefix = self.common_prefix.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let (tx, rx) = mpsc::channel(40960);

            tokio::spawn(async move {
                const BATCH_SIZE: usize = 4096;

                for path in paths {
                    let relative_name = generate_relative_name(&path, &common_prefix);

                    tx.send(DataItem::FileStart(relative_name)).await.unwrap();

                    let file = File::open(&path).await.unwrap();

                    let reader = BufReader::new(file);

                    let mut lines = reader.lines();

                    let mut batch: Vec<Cow<'static, str>> = Vec::with_capacity(BATCH_SIZE);

                    while let Some(line) = lines.next_line().await.unwrap() {
                        if line.trim().is_empty() {
                            continue;
                        }

                        let parsed: T = from_str(&line).unwrap();

                        let result = converter(parsed);

                        batch.push(result);

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
