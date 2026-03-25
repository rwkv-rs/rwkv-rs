use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
};

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};

use super::stats::{QueuePerfHistory, new_perf_history};
use crate::{
    cores::{
        forward::{ModelForward, TokenIdLogprobsConfig, sampling::SamplingConfig},
        queue::{GuidedDecodingConfig, Queue, QueueEvent, QueueItem, snapshot_perf_history},
    },
    dtos::health::QueuePerfSample,
};

#[derive(Clone, Debug)]
pub struct QueueHandle {
    tx: mpsc::Sender<QueueCommand>,
    pub pending: Arc<AtomicUsize>,
    accepting: Arc<AtomicBool>,
    pub device_id: u32,
    pub weights_path: String,
    pub max_batch_size: usize,
    pub tokenizer: Arc<Tokenizer>,
    perf_history: QueuePerfHistory,
}

#[derive(Clone, Debug)]
pub struct QueueSubmitRequest {
    pub prompt: String,
    pub sampling_config: SamplingConfig,
    pub token_logprobs_config: Option<TokenIdLogprobsConfig>,
    pub stop_suffixes: Vec<String>,
    pub guided_decoding_config: Option<GuidedDecodingConfig>,
}

#[derive(Debug)]
enum QueueCommand {
    Submit {
        request: QueueSubmitRequest,
        reply: oneshot::Sender<mpsc::Receiver<QueueEvent>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

impl QueueHandle {
    pub fn load_score(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire)
    }

    pub fn begin_drain(&self) {
        self.accepting.store(false, Ordering::Release);
    }

    pub fn perf_samples(&self) -> Vec<QueuePerfSample> {
        snapshot_perf_history(&self.perf_history)
    }

    pub async fn submit(&self, request: QueueSubmitRequest) -> mpsc::Receiver<QueueEvent> {
        assert!(
            self.is_accepting(),
            "queue {} is not accepting",
            self.device_id
        );
        self.pending.fetch_add(1, Ordering::Relaxed);

        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(QueueCommand::Submit {
                request,
                reply: reply_tx,
            })
            .await
            .unwrap();
        reply_rx.await.unwrap()
    }

    pub async fn shutdown(&self) {
        self.begin_drain();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(QueueCommand::Shutdown { reply: reply_tx })
            .await
            .unwrap();
        reply_rx.await.unwrap();
    }
}

pub fn spawn_queue_worker(
    model_forward: Box<dyn ModelForward>,
    tokenizer: Arc<Tokenizer>,
    max_batch_size: usize,
    paragraph_len: usize,
    device_id: u32,
    weights_path: String,
) -> QueueHandle {
    let pending = Arc::new(AtomicUsize::new(0));
    let accepting = Arc::new(AtomicBool::new(true));
    let perf_history = new_perf_history();
    let (tx, mut rx) = mpsc::channel(1024);
    let pending_for_thread = Arc::clone(&pending);
    let accepting_for_thread = Arc::clone(&accepting);
    let worker_tokenizer = Arc::clone(&tokenizer);
    let perf_history_for_thread = Arc::clone(&perf_history);

    thread::spawn(move || {
        let mut queue = Queue::new(
            model_forward,
            Arc::clone(&worker_tokenizer),
            max_batch_size,
            paragraph_len,
            perf_history_for_thread,
        );
        let mut active_items = 0usize;
        let mut next_item_id = 1usize;
        let mut shutdown_reply = None;

        loop {
            if active_items == 0 {
                let message = rx.blocking_recv();
                let Some(message) = message else {
                    break;
                };
                match message {
                    QueueCommand::Submit { request, reply } => {
                        let (completions_tx, completions_rx) = mpsc::channel(64);
                        let context_tokens_for_step =
                            tokenize_prompt(&worker_tokenizer, &request.prompt, paragraph_len);
                        let item = QueueItem::new(
                            context_tokens_for_step,
                            request.sampling_config,
                            request.token_logprobs_config,
                            request.stop_suffixes,
                            completions_tx,
                            request.guided_decoding_config,
                        );
                        queue.push(next_item_id, item).unwrap();
                        next_item_id += 1;
                        active_items += 1;
                        let _ = reply.send(completions_rx);
                    }
                    QueueCommand::Shutdown { reply } => {
                        accepting_for_thread.store(false, Ordering::Release);
                        shutdown_reply = Some(reply);
                    }
                }
            }

            while let Ok(message) = rx.try_recv() {
                match message {
                    QueueCommand::Submit { request, reply } => {
                        let (completions_tx, completions_rx) = mpsc::channel(64);
                        let context_tokens_for_step =
                            tokenize_prompt(&worker_tokenizer, &request.prompt, paragraph_len);
                        let item = QueueItem::new(
                            context_tokens_for_step,
                            request.sampling_config,
                            request.token_logprobs_config,
                            request.stop_suffixes,
                            completions_tx,
                            request.guided_decoding_config,
                        );
                        queue.push(next_item_id, item).unwrap();
                        next_item_id += 1;
                        active_items += 1;
                        let _ = reply.send(completions_rx);
                    }
                    QueueCommand::Shutdown { reply } => {
                        accepting_for_thread.store(false, Ordering::Release);
                        shutdown_reply = Some(reply);
                    }
                }
            }

            if active_items > 0 {
                let finished_items = queue.run_once();
                if finished_items > 0 {
                    debug_assert!(finished_items <= active_items);
                    active_items -= finished_items;
                    pending_for_thread.fetch_sub(finished_items, Ordering::Relaxed);
                }
            }

            if active_items == 0 {
                if let Some(reply) = shutdown_reply.take() {
                    let _ = reply.send(());
                    break;
                }
            }
        }
    });

    QueueHandle {
        tx,
        pending,
        accepting,
        device_id,
        weights_path,
        max_batch_size,
        tokenizer,
        perf_history,
    }
}

fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str, paragraph_len: usize) -> Vec<i32> {
    let mut context_tokens_for_step = vec![0];
    context_tokens_for_step.extend(tokenizer.encode(prompt, false).into_iter().map(i32::from));

    while context_tokens_for_step.len() % paragraph_len != 0 {
        context_tokens_for_step.insert(0, 0);
    }

    context_tokens_for_step
}
