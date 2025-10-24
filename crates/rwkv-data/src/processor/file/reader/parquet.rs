use std::{
    borrow::Cow,
    fs::File,
    future::Future,
    marker::PhantomData,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Row,
};
use tokio::sync::mpsc::{Receiver, channel};

use crate::processor::file::{DataItem, Reader, find_common_prefix, generate_relative_name};

pub trait FromParquetRow: Sized {
    fn from_row(row: &Row) -> Self;
}

pub struct ParquetReader<T, F>
where
    T: FromParquetRow + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    paths: Vec<PathBuf>,
    common_prefix: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> ParquetReader<T, F>
where
    T: FromParquetRow + Send + 'static,
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
}

impl<T, F> Reader for ParquetReader<T, F>
where
    T: FromParquetRow + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>> {
        let paths = self.paths.clone();

        let common_prefix = self.common_prefix.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let (tx, rx) = channel(1000);

            tokio::task::spawn_blocking(move || {
                const BATCH_SIZE: usize = 4096;

                for path in paths {
                    let relative_name = generate_relative_name(&path, &common_prefix);

                    if tokio::runtime::Handle::current()
                        .block_on(tx.send(DataItem::FileStart(relative_name)))
                        .is_err()
                    {
                        break;
                    }

                    let file = File::open(&path).unwrap();

                    let reader = SerializedFileReader::new(file).unwrap();

                    let mut iter = reader.get_row_iter(None).unwrap();

                    let mut batch: Vec<Cow<'static, str>> = Vec::with_capacity(BATCH_SIZE);

                    while let Some(row) = iter.next() {
                        let row = row.unwrap();

                        let parsed = T::from_row(&row);

                        let converted = converter(parsed);

                        batch.push(converted);

                        if batch.len() >= BATCH_SIZE {
                            let to_send = std::mem::take(&mut batch);

                            if tokio::runtime::Handle::current()
                                .block_on(tx.send(DataItem::DataBatch(to_send)))
                                .is_err()
                            {
                                return;
                            }

                            batch = Vec::with_capacity(BATCH_SIZE);
                        }
                    }

                    if !batch.is_empty() {
                        let to_send = std::mem::take(&mut batch);

                        let _ = tokio::runtime::Handle::current()
                            .block_on(tx.send(DataItem::DataBatch(to_send)));
                    }
                }
            });

            rx
        })
    }
}
