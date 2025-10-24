use std::{
    borrow::Cow,
    future::Future,
    marker::PhantomData,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use serde::Serialize;
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
    sync::mpsc::Receiver,
};

use crate::processor::file::{DataItem, Writer};

pub struct JsonWriter<T, F>
where
    T: Serialize + Send + 'static,
    F: Fn(Cow<'static, str>) -> Option<T> + Send + Sync + 'static,
{
    output_dir: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> JsonWriter<T, F>
where
    T: Serialize + Send + 'static,
    F: Fn(Cow<'static, str>) -> Option<T> + Send + Sync + 'static,
{
    pub fn new(output_dir: &Path, converter: F) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            converter: Arc::new(converter),
            phantom: PhantomData,
        }
    }
}

impl<T, F> Writer for JsonWriter<T, F>
where
    T: Serialize + Send + 'static,
    F: Fn(Cow<'static, str>) -> Option<T> + Send + Sync + 'static,
{
    fn run(&self, mut rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let output_dir = self.output_dir.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let mut current_writer: Option<BufWriter<File>> = None;

            while let Some(item) = rx.recv().await {
                match item {
                    DataItem::FileStart(relative_name) => {
                        if let Some(mut writer) = current_writer.take() {
                            writer.flush().await.unwrap();
                        }

                        let output_filename = format!("{}.jsonl", relative_name);

                        let output_path = output_dir.join(output_filename);

                        if let Some(parent) = output_path.parent() {
                            tokio::fs::create_dir_all(parent).await.unwrap();
                        }

                        let file = File::create(output_path).await.unwrap();

                        current_writer = Some(BufWriter::new(file));
                    },
                    DataItem::DataBatch(batch) => {
                        if let Some(writer) = current_writer.as_mut() {
                            for data in batch {
                                if let Some(parsed) = converter(data) {
                                    let json_line = sonic_rs::to_string(&parsed).unwrap();

                                    writer.write_all(json_line.as_bytes()).await.unwrap();

                                    writer.write_all(b"\n").await.unwrap();
                                }
                            }
                        }
                    },
                }
            }

            if let Some(mut writer) = current_writer {
                writer.flush().await.unwrap();
            }
        })
    }
}
