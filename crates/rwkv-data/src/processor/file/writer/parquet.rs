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
    column::writer::get_typed_column_writer_mut,
    data_type::{ByteArray, ByteArrayType},
    file::writer::SerializedFileWriter,
    schema::parser::parse_message_type,
};
use tokio::sync::{mpsc, mpsc::Receiver};

use crate::processor::file::{DataItem, Writer};

pub trait ToParquetRow {
    fn to_row_data(&self) -> Vec<String>;

    fn schema() -> &'static str;
}

pub struct ParquetWriter<T, F>
where
    T: ToParquetRow + Send + 'static,
    F: Fn(Cow<'static, str>) -> T + Send + Sync + 'static,
{
    output_dir: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> ParquetWriter<T, F>
where
    T: ToParquetRow + Send + 'static,
    F: Fn(Cow<'static, str>) -> T + Send + Sync + 'static,
{
    pub fn new(output_dir: &Path, converter: F) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            converter: Arc::new(converter),
            phantom: PhantomData,
        }
    }
}

impl<T, F> Writer for ParquetWriter<T, F>
where
    T: ToParquetRow + Send + 'static,
    F: Fn(Cow<'static, str>) -> T + Send + Sync + 'static,
{
    fn run(&self, mut rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let output_dir = self.output_dir.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let mut current_writer_task: Option<tokio::task::JoinHandle<()>> = None;

            let mut current_data_tx: Option<tokio::sync::mpsc::Sender<Vec<String>>> = None;

            while let Some(item) = rx.recv().await {
                match item {
                    DataItem::FileStart(relative_name) => {
                        // 关闭当前文件
                        if let Some(tx) = current_data_tx.take() {
                            drop(tx);
                        }

                        if let Some(task) = current_writer_task.take() {
                            let _ = task.await;
                        }

                        // 创建新文件
                        let output_filename = format!("{}.parquet", relative_name);

                        let output_path = output_dir.join(output_filename);

                        if let Some(parent) = output_path.parent() {
                            tokio::fs::create_dir_all(parent).await.unwrap();
                        }

                        let (data_tx, mut data_rx) = mpsc::channel::<Vec<String>>(40960);

                        current_data_tx = Some(data_tx);

                        let write_path = output_path.to_string_lossy().to_string();

                        current_writer_task = Some(tokio::task::spawn_blocking(move || {
                            let schema = Arc::new(parse_message_type(T::schema()).unwrap());

                            let file = File::create(&write_path).unwrap();

                            let mut writer =
                                SerializedFileWriter::new(file, schema, Default::default())
                                    .unwrap();

                            let mut row_group_writer = writer.next_row_group().unwrap();

                            if let Some(mut col_writer) = row_group_writer.next_column().unwrap() {
                                let typed_writer = get_typed_column_writer_mut::<ByteArrayType>(
                                    col_writer.untyped(),
                                );

                                let mut batch_values = Vec::new();

                                while let Some(row_data) = data_rx.blocking_recv() {
                                    for value in row_data {
                                        batch_values.push(ByteArray::from(value.as_str()));
                                    }

                                    if batch_values.len() >= 1000 {
                                        typed_writer
                                            .write_batch(&batch_values, None, None)
                                            .unwrap();

                                        batch_values.clear();
                                    }
                                }

                                if !batch_values.is_empty() {
                                    typed_writer.write_batch(&batch_values, None, None).unwrap();
                                }

                                col_writer.close().unwrap();
                            }

                            row_group_writer.close().unwrap();

                            writer.close().unwrap();
                        }));
                    }
                    DataItem::DataBatch(batch) => {
                        if let Some(tx) = &current_data_tx {
                            for data in batch {
                                let parsed = converter(data);

                                let row_data = parsed.to_row_data();

                                if tx.send(row_data).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }

            // 关闭最后一个文件
            if let Some(tx) = current_data_tx {
                drop(tx);
            }

            if let Some(task) = current_writer_task {
                let _ = task.await;
            }
        })
    }
}
