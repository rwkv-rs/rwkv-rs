use std::{
    borrow::Cow,
    future::Future,
    marker::PhantomData,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use csv_async::AsyncWriterBuilder;
use tokio::{fs::File, sync::mpsc::Receiver};

use crate::processor::file::{DataItem, Writer};

pub trait ToCsvRecord {
    fn to_csv_fields(&self) -> Vec<String>;
}

pub struct CsvWriter<T, F>
where
    T: ToCsvRecord + Send + 'static,
    F: Fn(Cow<'static, str>) -> T + Send + Sync + 'static,
{
    output_dir: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> CsvWriter<T, F>
where
    T: ToCsvRecord + Send + 'static,
    F: Fn(Cow<'static, str>) -> T + Send + Sync + 'static,
{
    pub fn new(output_dir: &Path, converter: F) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
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

    fn get_extension(delimiter: u8) -> &'static str {
        match delimiter {
            b'\t' => "tsv",
            _ => "csv",
        }
    }
}

impl<T, F> Writer for CsvWriter<T, F>
where
    T: ToCsvRecord + Send + 'static,
    F: Fn(Cow<'static, str>) -> T + Send + Sync + 'static,
{
    fn run(&self, mut rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let output_dir = self.output_dir.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let mut current_writer: Option<csv_async::AsyncWriter<File>> = None;

            while let Some(item) = rx.recv().await {
                match item {
                    DataItem::FileStart(relative_name) => {
                        if let Some(mut writer) = current_writer.take() {
                            let _ = writer.flush().await;
                        }

                        let mut output_path = output_dir.join(&relative_name);

                        let delimiter = Self::get_delimiter(&output_path);

                        if output_path.extension().is_none() {
                            output_path.set_extension(Self::get_extension(delimiter));
                        }

                        if let Some(parent) = output_path.parent() {
                            tokio::fs::create_dir_all(parent).await.unwrap();
                        }

                        let file = File::create(output_path).await.unwrap();

                        let writer = AsyncWriterBuilder::new()
                            .delimiter(delimiter)
                            .create_writer(file);

                        current_writer = Some(writer);
                    },
                    DataItem::DataBatch(batch) => {
                        if let Some(writer) = current_writer.as_mut() {
                            for data in batch {
                                let parsed = converter(data);

                                let fields = parsed.to_csv_fields();

                                writer.write_record(&fields).await.unwrap();
                            }
                        }
                    },
                }
            }

            if let Some(mut writer) = current_writer {
                let _ = writer.flush().await;
            }
        })
    }
}
