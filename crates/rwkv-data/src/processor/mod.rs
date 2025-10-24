// 我希望的用户写法
// processor = Processor::new(
//   reader = JsonReader::new("读取的jsonl路径"),
//   writer = JsonWriter::new("写入的jsonl路径"),
//   steps = [
//      balabala
//   ],
// );
// processor.run();
pub mod file;
pub mod pool;
pub mod stream;

use std::{
    borrow::Cow,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender},
};

use crate::processor::file::{DataItem, Reader, Writer};

const CHANNEL_BUFFER_SIZE: usize = 4096;

pub struct StepStats {
    start_time: Instant,
    total_input_count: u64,
    total_output_count: u64,
    total_input_length: u64,
    total_output_length: u64,
}

impl StepStats {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_input_count: 0,
            total_output_count: 0,
            total_input_length: 0,
            total_output_length: 0,
        }
    }

    pub fn record(&mut self, input_len: usize, output_len: Option<usize>) {
        self.total_input_count += 1;

        self.total_input_length += input_len as u64;

        if let Some(len) = output_len {
            self.total_output_count += 1;

            self.total_output_length += len as u64;
        }
    }

    pub fn get_message(&self) -> String {
        let elapsed = self.start_time.elapsed();

        let rate = if elapsed.as_secs() > 0 {
            self.total_input_count / elapsed.as_secs()
        } else {
            self.total_input_count
        };

        let avg_input_len = if self.total_input_count > 0 {
            self.total_input_length / self.total_input_count
        } else {
            0
        };

        let avg_output_len = if self.total_output_count > 0 {
            self.total_output_length / self.total_output_count
        } else {
            0
        };

        format!(
            "Inputs: {} | Outputs: {} | Rate: {}/s | Avg len: {} -> {}",
            self.total_input_count, self.total_output_count, rate, avg_input_len, avg_output_len
        )
    }

    pub fn snapshot(&self, step_name: &str, step_index: usize) -> StepStatsSnapshot {
        StepStatsSnapshot {
            step_name: step_name.to_string(),
            step_index,
            total_input_count: self.total_input_count,
            total_output_count: self.total_output_count,
            total_input_length: self.total_input_length,
            total_output_length: self.total_output_length,
            elapsed: self.start_time.elapsed(),
        }
    }
}

#[derive(Clone)]

pub struct StepStatsSnapshot {
    step_name: String,
    step_index: usize,
    total_input_count: u64,
    total_output_count: u64,
    total_input_length: u64,
    total_output_length: u64,
    elapsed: Duration,
}

impl StepStatsSnapshot {
    fn summary(&self) -> String {
        let elapsed_secs = self.elapsed.as_secs_f64().max(1e-6);

        let rate = (self.total_input_count as f64 / elapsed_secs).round() as u64;

        let avg_input_len = if self.total_input_count > 0 {
            self.total_input_length / self.total_input_count
        } else {
            0
        };

        let avg_output_len = if self.total_output_count > 0 {
            self.total_output_length / self.total_output_count
        } else {
            0
        };

        format!(
            "{} | Inputs: {} | Outputs: {} | Rate: {}/s | Avg len: {} -> {}",
            self.step_name,
            self.total_input_count,
            self.total_output_count,
            rate,
            avg_input_len,
            avg_output_len
        )
    }
}

pub struct Processor<R: Reader, W: Writer> {
    reader: R,
    steps: Vec<Arc<dyn Step>>,
    writer: W,
    multi_progress: MultiProgress,
    stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
}

impl<R: Reader, W: Writer> Processor<R, W> {
    pub fn new(reader: R, steps: Vec<Arc<dyn Step>>, writer: W) -> Self {
        Self {
            reader,
            steps,
            writer,
            multi_progress: MultiProgress::new(),
            stats_collector: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn run(&self) {
        let style = ProgressStyle::with_template("[{elapsed_precise}] {prefix} | {msg}").unwrap();

        let mut rx = self.reader.run().await;

        for (index, step) in self.steps.iter().enumerate() {
            let progress = self.multi_progress.add(ProgressBar::new_spinner());

            progress.set_style(style.clone());

            progress.set_prefix(step.name().to_string());

            progress.set_message("Starting...");

            rx = StepExecutor::new(
                Arc::clone(step),
                progress,
                index,
                Arc::clone(&self.stats_collector),
            )
            .run(rx);
        }

        self.writer.run(rx).await;

        self.print_step_summaries().await;
    }

    async fn print_step_summaries(&self) {
        let mut guard = self.stats_collector.lock().await;

        guard.sort_by_key(|snapshot| snapshot.step_index);

        for snapshot in guard.iter() {
            println!("{}", snapshot.summary());
        }
    }
}

#[derive(Debug)]

pub enum StepOutcome {
    Keep(Cow<'static, str>),
    Exclude(Cow<'static, str>),
}

pub trait Step: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn batch_size(&self) -> NonZeroUsize;

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static>;

    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome>;
}

struct StepExecutor {
    step: Arc<dyn Step>,
    progress: ProgressBar,
    batch_size: usize,
    step_index: usize,
    stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
}

impl StepExecutor {
    fn new(
        step: Arc<dyn Step>,
        progress: ProgressBar,
        step_index: usize,
        stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
    ) -> Self {
        let batch_size = step.batch_size().get();

        Self {
            step,
            progress,
            batch_size,
            step_index,
            stats_collector,
        }
    }

    fn run(self, input: Receiver<DataItem>) -> Receiver<DataItem> {
        let (tx_next, rx_next) = mpsc::channel(CHANNEL_BUFFER_SIZE);

        let (tx_exclusion, rx_exclusion) = mpsc::channel(CHANNEL_BUFFER_SIZE);

        let step = self.step;

        let progress = self.progress;

        let batch_size = self.batch_size;

        let step_index = self.step_index;

        let stats_collector = Arc::clone(&self.stats_collector);

        let writer = step.exclusion_writer();

        tokio::spawn(async move {
            writer.run(rx_exclusion).await;
        });

        tokio::spawn(async move {
            StepWorker::new(
                step,
                progress,
                batch_size,
                step_index,
                tx_next,
                tx_exclusion,
                stats_collector,
            )
            .run(input)
            .await;
        });

        rx_next
    }
}

struct StepWorker {
    step: Arc<dyn Step>,
    progress: ProgressBar,
    batch_size: usize,
    stats: StepStats,
    step_index: usize,
    tx_next: Sender<DataItem>,
    tx_exclusion: Sender<DataItem>,
    stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
}

impl StepWorker {
    fn new(
        step: Arc<dyn Step>,
        progress: ProgressBar,
        batch_size: usize,
        step_index: usize,
        tx_next: Sender<DataItem>,
        tx_exclusion: Sender<DataItem>,
        stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
    ) -> Self {
        Self {
            step,
            progress,
            batch_size,
            stats: StepStats::new(),
            step_index,
            tx_next,
            tx_exclusion,
            stats_collector,
        }
    }

    async fn run(mut self, mut input: Receiver<DataItem>) {
        let mut data_batch: Vec<(Cow<'static, str>, usize)> = Vec::with_capacity(self.batch_size);

        while let Some(item) = input.recv().await {
            match item {
                DataItem::DataBatch(mut batch) => {
                    for data in batch.drain(..) {
                        let len = data.len();

                        data_batch.push((data, len));

                        if data_batch.len() >= self.batch_size {
                            if self.flush(&mut data_batch).await.is_err() {
                                return;
                            }
                        }
                    }
                },
                DataItem::FileStart(path) => {
                    if !data_batch.is_empty() && self.flush(&mut data_batch).await.is_err() {
                        break;
                    }

                    if self
                        .forward_control(DataItem::FileStart(path))
                        .await
                        .is_err()
                    {
                        break;
                    }
                },
            }
        }

        let _ = self.flush(&mut data_batch).await;

        let snapshot = self.stats.snapshot(self.step.name(), self.step_index);

        let mut guard = self.stats_collector.lock().await;

        guard.push(snapshot);

        self.progress
            .finish_with_message(format!("✓ {} completed", self.step.name()));
    }

    async fn flush(&mut self, data_batch: &mut Vec<(Cow<'static, str>, usize)>) -> Result<(), ()> {
        if data_batch.is_empty() {
            return Ok(());
        }

        let items = std::mem::take(data_batch);

        data_batch.reserve(self.batch_size);

        let mut input_lengths = Vec::with_capacity(items.len());

        let inputs: Vec<_> = items
            .into_iter()
            .map(|(data, len)| {
                input_lengths.push(len);

                data
            })
            .collect();

        let results = self.step.process_batch(inputs);

        if results.len() != input_lengths.len() {
            // 设计保证：返回结果必须与输入等长
            return Err(());
        }

        let mut keep_batch: Vec<Cow<'static, str>> = Vec::with_capacity(results.len());

        let mut exclude_batch: Vec<Cow<'static, str>> = Vec::with_capacity(results.len());

        for (input_len, outcome) in input_lengths.into_iter().zip(results.into_iter()) {
            match outcome {
                StepOutcome::Keep(output) => {
                    let output_len = output.len();

                    self.stats.record(input_len, Some(output_len));

                    self.progress.set_message(self.stats.get_message());

                    self.progress.inc(1);

                    keep_batch.push(output);
                },
                StepOutcome::Exclude(output) => {
                    self.stats.record(input_len, None);

                    self.progress.set_message(self.stats.get_message());

                    self.progress.inc(1);

                    exclude_batch.push(output);
                },
            }
        }

        if !keep_batch.is_empty()
            && self
                .tx_next
                .send(DataItem::DataBatch(keep_batch))
                .await
                .is_err()
        {
            return Err(());
        }

        if !exclude_batch.is_empty()
            && self
                .tx_exclusion
                .send(DataItem::DataBatch(exclude_batch))
                .await
                .is_err()
        {
            return Err(());
        }

        Ok(())
    }

    async fn forward_control(&mut self, item: DataItem) -> Result<(), ()> {
        let clone_for_exclusion = match &item {
            DataItem::FileStart(path) => DataItem::FileStart(path.clone()),
            DataItem::DataBatch(_) => unreachable!("控制消息不应为 DataBatch"),
        };

        if self.tx_next.send(item).await.is_err() {
            return Err(());
        }

        if self.tx_exclusion.send(clone_for_exclusion).await.is_err() {
            return Err(());
        }

        Ok(())
    }
}
