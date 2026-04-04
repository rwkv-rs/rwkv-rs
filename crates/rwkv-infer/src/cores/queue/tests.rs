use std::{
    collections::HashMap,
    fs,
    panic::{self, AssertUnwindSafe},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{
    END_TOKEN_ID,
    Queue,
    QueueEvent,
    QueueFinishMeta,
    QueueFinishReason,
    QueueItem,
    QueueItemStatus,
    stats::new_perf_history,
};
use crate::cores::{
    forward::{ModelForward, StepMode, TokenId},
    queue::guided_decode::{
        GuidedDecodingConfig,
        GuidedDecodingStatus,
        GuidedMatcherState,
        GuidedPreparedState,
    },
};

#[derive(Default)]
struct DummyModelForward {
    choose_first_allowed_token: bool,
    fixed_token_id: i32,
    sample_guided_token_masks: Arc<Mutex<Vec<Vec<Option<Vec<i32>>>>>>,
    guided_token_masks_by_batch: HashMap<usize, Vec<i32>>,
}

impl ModelForward for DummyModelForward {
    fn step(
        &mut self,
        batch_ids: &[usize],
        _contexts: &[&[i32]],
        _context_masks: &[&[u8]],
        mode: StepMode<'_>,
    ) -> Option<Vec<TokenId>> {
        match mode {
            StepMode::PrefillNoOutput => None,
            StepMode::Sample {
                has_masked_guided_token,
                ..
            } => {
                let sampled_guided_token_masks = batch_ids
                    .iter()
                    .map(|batch_id| self.guided_token_masks_by_batch.get(batch_id).cloned())
                    .collect::<Vec<_>>();
                if !has_masked_guided_token {
                    assert!(
                        sampled_guided_token_masks.iter().all(Option::is_none),
                        "sample step declared no masked guided token but forward still saw masked rows"
                    );
                }
                self.sample_guided_token_masks
                    .lock()
                    .unwrap()
                    .push(sampled_guided_token_masks.clone());

                Some(
                    batch_ids
                        .iter()
                        .copied()
                        .zip(sampled_guided_token_masks.iter())
                        .map(|(batch_index, guided_token_mask)| {
                            let token_id = if self.choose_first_allowed_token {
                                guided_token_mask
                                    .as_deref()
                                    .and_then(bitmask_first_allowed_token_id)
                                    .unwrap_or(END_TOKEN_ID)
                            } else {
                                self.fixed_token_id
                            };

                            TokenId {
                                batch_index,
                                token_id,
                                logprob: None,
                                finish_after_token: false,
                            }
                        })
                        .collect(),
                )
            }
        }
    }

    fn set_guided_token_mask_row(&mut self, batch_index: usize, token_mask: Option<&[i32]>) {
        if let Some(token_mask) = token_mask {
            self.guided_token_masks_by_batch
                .insert(batch_index, token_mask.to_vec());
        } else {
            self.guided_token_masks_by_batch.remove(&batch_index);
        }
    }

    fn reset(&mut self, batch_index: usize) {
        self.guided_token_masks_by_batch.remove(&batch_index);
    }
}

#[test]
fn prefill_without_output_does_not_wait_for_guided_step() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        2,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0, 1, 0, 1],
                Default::default(),
                None,
                vec![],
                completions_tx,
                Some(test_guided_decoding_config()),
            ),
        )
        .expect("push guided item");

    assert_eq!(queue.collect_step_item_ids(), vec![1]);
}

#[test]
fn prefill_waits_for_guided_step() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        2,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0, 1],
                Default::default(),
                None,
                vec![],
                completions_tx,
                Some(test_guided_decoding_config()),
            ),
        )
        .expect("push guided item");

    assert!(queue.collect_step_item_ids().is_empty());

    for _ in 0..100 {
        queue.drain_guided_decode_results();
        queue.update_batch_status();

        if queue.items.get(&1).is_some_and(|item| {
            matches!(item.guided_decoding_status, GuidedDecodingStatus::Ready(_))
        }) {
            break;
        }

        thread::sleep(Duration::from_millis(1));
    }

    assert!(queue.items.get(&1).is_some_and(|item| {
        matches!(item.guided_decoding_status, GuidedDecodingStatus::Ready(_))
    }));
    assert_eq!(queue.collect_step_item_ids(), vec![1]);
}

#[test]
fn step_passes_guided_token_masks_to_sample() {
    let sample_guided_token_masks = Arc::new(Mutex::new(Vec::new()));
    let mut queue = Queue::new(
        Box::new(DummyModelForward {
            choose_first_allowed_token: false,
            fixed_token_id: END_TOKEN_ID,
            sample_guided_token_masks: Arc::clone(&sample_guided_token_masks),
            guided_token_masks_by_batch: HashMap::new(),
        }),
        test_tokenizer(),
        4,
        2,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0, 1],
                Default::default(),
                None,
                vec![],
                completions_tx,
                None,
            ),
        )
        .expect("push item");

    let prepared_state = GuidedPreparedState::masked(
        test_guided_matcher_state(&queue),
        vec![123].into_boxed_slice(),
    );
    let item = queue.items.get_mut(&1).expect("queue item");
    item.guided_decoding_status = GuidedDecodingStatus::Ready(prepared_state);

    queue.update_batch_status();
    let item_ids = queue.collect_step_item_ids();
    assert_eq!(item_ids, vec![1]);

    let new_tokens = queue.step(&item_ids).expect("sample output");
    assert_eq!(new_tokens.len(), 1);

    let sample_guided_token_masks = sample_guided_token_masks.lock().unwrap().clone();
    assert_eq!(sample_guided_token_masks, vec![vec![Some(vec![123])]]);
}

#[test]
fn step_passes_all_allowed_guided_step_as_none_to_sample() {
    let sample_guided_token_masks = Arc::new(Mutex::new(Vec::new()));
    let mut queue = Queue::new(
        Box::new(DummyModelForward {
            choose_first_allowed_token: false,
            fixed_token_id: END_TOKEN_ID,
            sample_guided_token_masks: Arc::clone(&sample_guided_token_masks),
            guided_token_masks_by_batch: HashMap::new(),
        }),
        test_tokenizer(),
        4,
        2,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0, 1],
                Default::default(),
                None,
                vec![],
                completions_tx,
                None,
            ),
        )
        .expect("push item");

    let prepared_state = GuidedPreparedState::all_allowed(test_guided_matcher_state(&queue));
    let item = queue.items.get_mut(&1).expect("queue item");
    item.guided_decoding_status = GuidedDecodingStatus::Ready(prepared_state);

    queue.update_batch_status();
    let item_ids = queue.collect_step_item_ids();
    assert_eq!(item_ids, vec![1]);

    let new_tokens = queue.step(&item_ids).expect("sample output");
    assert_eq!(new_tokens.len(), 1);

    let sample_guided_token_masks = sample_guided_token_masks.lock().unwrap().clone();
    assert_eq!(sample_guided_token_masks, vec![vec![None]]);
}

#[test]
fn slot_reuse_resets_persistent_guided_mask_state() {
    let sample_guided_token_masks = Arc::new(Mutex::new(Vec::new()));
    let mut queue = Queue::new(
        Box::new(DummyModelForward {
            choose_first_allowed_token: false,
            fixed_token_id: END_TOKEN_ID,
            sample_guided_token_masks: Arc::clone(&sample_guided_token_masks),
            guided_token_masks_by_batch: HashMap::new(),
        }),
        test_tokenizer(),
        1,
        1,
        new_perf_history(),
    );
    let (first_completions_tx, _first_completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                first_completions_tx,
                None,
            ),
        )
        .expect("push first item");

    let first_guided_token_mask = build_guided_token_mask(queue.guided_vocab_size, &[1]).to_vec();
    queue
        .items
        .get_mut(&1)
        .expect("queue item")
        .guided_decoding_status = GuidedDecodingStatus::Ready(GuidedPreparedState::masked(
        test_guided_matcher_state(&queue),
        first_guided_token_mask.clone().into_boxed_slice(),
    ));

    queue.update_batch_status();
    let first_item_ids = queue.collect_step_item_ids();
    assert_eq!(first_item_ids, vec![1]);
    let first_new_tokens = queue.step(&first_item_ids).expect("sample output");
    assert_eq!(first_new_tokens.len(), 1);

    queue.remove(&[1]);
    assert!(queue.items.is_empty());

    let (second_completions_tx, _second_completions_rx) = mpsc::channel(8);
    queue
        .push(
            2,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                second_completions_tx,
                None,
            ),
        )
        .expect("push second item");

    queue.update_batch_status();
    let second_item_ids = queue.collect_step_item_ids();
    assert_eq!(second_item_ids, vec![2]);
    let second_new_tokens = queue.step(&second_item_ids).expect("sample output");
    assert_eq!(second_new_tokens.len(), 1);

    let sample_guided_token_masks = sample_guided_token_masks.lock().unwrap().clone();
    assert_eq!(
        sample_guided_token_masks,
        vec![vec![Some(first_guided_token_mask)], vec![None]]
    );
}

#[test]
fn run_guided_item_end_to_end() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward {
            choose_first_allowed_token: true,
            fixed_token_id: END_TOKEN_ID,
            sample_guided_token_masks: Arc::new(Mutex::new(Vec::new())),
            guided_token_masks_by_batch: HashMap::new(),
        }),
        test_tokenizer(),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, mut completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                completions_tx,
                Some(test_guided_decoding_config()),
            ),
        )
        .expect("push guided item");

    queue.run();

    let mut completions_text = String::new();
    let mut saw_done = false;
    while let Ok(event) = completions_rx.try_recv() {
        match event {
            QueueEvent::Delta(delta) => completions_text.push_str(&delta.text),
            QueueEvent::Done(_) => saw_done = true,
        }
    }

    assert_eq!(completions_text, "{}");
    assert!(saw_done);
    assert!(queue.items.is_empty());
}

#[test]
fn late_guided_result_is_ignored_after_remove() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                completions_tx,
                Some(test_guided_decoding_config()),
            ),
        )
        .expect("push guided item");

    queue.remove(&[1]);
    for _ in 0..100 {
        queue.drain_guided_decode_results();
        thread::sleep(Duration::from_millis(1));
    }

    assert!(queue.items.is_empty());
}

#[test]
fn guided_mask_violation_panics_before_detokenize() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                completions_tx,
                None,
            ),
        )
        .expect("push item");

    let prepared_state = GuidedPreparedState::masked(
        test_guided_matcher_state(&queue),
        build_guided_token_mask(queue.guided_vocab_size, &[1]),
    );
    let item = queue.items.get_mut(&1).expect("queue item");
    item.batch_id = Some(0);
    item.status = QueueItemStatus::Decode(0);
    item.guided_decoding_status = GuidedDecodingStatus::Ready(prepared_state);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        queue.apply_output_tokens(
            &[1],
            vec![TokenId {
                batch_index: 0,
                token_id: 2,
                logprob: None,
                finish_after_token: false,
            }],
        );
    }));

    let panic_message = panic_message(result.expect_err("guided mask violation should panic"));
    assert!(panic_message.contains("sampler violated guided mask"));
    assert_eq!(
        queue
            .items
            .get(&1)
            .expect("queue item")
            .pending_detokenize_tasks,
        0
    );
}

#[test]
fn guided_mask_state_mismatch_panics_before_detokenize() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, _completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                completions_tx,
                None,
            ),
        )
        .expect("push item");

    let prepared_state = GuidedPreparedState::all_allowed(test_guided_matcher_state(&queue));
    let item = queue.items.get_mut(&1).expect("queue item");
    item.batch_id = Some(0);
    item.status = QueueItemStatus::Decode(0);
    item.guided_decoding_status = GuidedDecodingStatus::Ready(prepared_state);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        queue.apply_output_tokens(
            &[1],
            vec![TokenId {
                batch_index: 0,
                token_id: 3,
                logprob: None,
                finish_after_token: false,
            }],
        );
    }));

    let panic_message = panic_message(result.expect_err("guided state mismatch should panic"));
    assert!(panic_message.contains("guided mask/state mismatch"));
    assert_eq!(
        queue
            .items
            .get(&1)
            .expect("queue item")
            .pending_detokenize_tasks,
        0
    );
}

#[test]
fn stop_suffix_inside_token_emits_text_without_partial_token() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer_with_vocab("1 \"helloUser:\" 10\n"),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, mut completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec!["User:".to_string()],
                completions_tx,
                None,
            ),
        )
        .expect("push item");
    queue.items.get_mut(&1).expect("queue item").status = QueueItemStatus::Decode(1);

    queue
        .queue_detokenize_token(
            1,
            TokenId {
                batch_index: 0,
                token_id: 1,
                logprob: None,
                finish_after_token: false,
            },
        )
        .expect("queue detokenize");
    wait_for_detokenize(&mut queue, 1);

    match completions_rx.try_recv().expect("delta event") {
        QueueEvent::Delta(delta) => {
            assert_eq!(delta.text, "hello");
            assert!(delta.tokens.is_empty());
        }
        event => panic!("unexpected event: {event:?}"),
    }
    match completions_rx.try_recv().expect("done event") {
        QueueEvent::Done(meta) => {
            assert_eq!(
                meta,
                QueueFinishMeta {
                    reason: QueueFinishReason::Stop,
                    matched_stop_suffix: Some("User:".to_string()),
                    matched_stop_suffix_index: Some(0),
                    generated_tokens: 1,
                }
            );
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert!(queue.items.is_empty());
}

#[test]
fn utf8_multibyte_sequence_waits_until_complete() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer_with_vocab("1 b\"\\xE4\" 1\n2 b\"\\xB8\" 1\n3 b\"\\x96\" 1\n"),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, mut completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec![],
                completions_tx,
                None,
            ),
        )
        .expect("push item");

    for token_id in [1, 2] {
        queue
            .queue_detokenize_token(
                1,
                TokenId {
                    batch_index: 0,
                    token_id,
                    logprob: None,
                    finish_after_token: false,
                },
            )
            .expect("queue detokenize");
        wait_for_detokenize(&mut queue, 1);
        assert!(completions_rx.try_recv().is_err());
    }

    queue
        .queue_detokenize_token(
            1,
            TokenId {
                batch_index: 0,
                token_id: 3,
                logprob: None,
                finish_after_token: false,
            },
        )
        .expect("queue detokenize");
    wait_for_detokenize(&mut queue, 1);

    match completions_rx.try_recv().expect("delta event") {
        QueueEvent::Delta(delta) => {
            assert_eq!(delta.text, "世");
            assert_eq!(delta.tokens.len(), 3);
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn finish_item_flushes_stop_tail_without_new_detokenize_result() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer_with_vocab("1 \"abc\" 3\n"),
        4,
        1,
        new_perf_history(),
    );
    let (completions_tx, mut completions_rx) = mpsc::channel(8);

    queue
        .push(
            1,
            QueueItem::new(
                vec![0],
                Default::default(),
                None,
                vec!["abcd".to_string()],
                completions_tx,
                None,
            ),
        )
        .expect("push item");
    queue.items.get_mut(&1).expect("queue item").status = QueueItemStatus::Decode(1);

    queue
        .queue_detokenize_token(
            1,
            TokenId {
                batch_index: 0,
                token_id: 1,
                logprob: None,
                finish_after_token: false,
            },
        )
        .expect("queue detokenize");
    wait_for_detokenize(&mut queue, 1);
    assert!(completions_rx.try_recv().is_err());

    let item = queue.items.get_mut(&1).expect("queue item");
    item.finish_meta = Some(QueueFinishMeta {
        reason: QueueFinishReason::Length,
        matched_stop_suffix: None,
        matched_stop_suffix_index: None,
        generated_tokens: 1,
    });
    item.status = QueueItemStatus::Finished;

    assert!(queue.finish_item_if_ready(1));

    match completions_rx.try_recv().expect("delta event") {
        QueueEvent::Delta(delta) => {
            assert_eq!(delta.text, "abc");
            assert_eq!(delta.tokens.len(), 1);
        }
        event => panic!("unexpected event: {event:?}"),
    }
    match completions_rx.try_recv().expect("done event") {
        QueueEvent::Done(meta) => {
            assert_eq!(meta.reason, QueueFinishReason::Length);
            assert_eq!(meta.generated_tokens, 1);
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

fn test_guided_decoding_config() -> GuidedDecodingConfig {
    GuidedDecodingConfig {
        schema_json: r#"{"type":"object","properties":{},"additionalProperties":false}"#
            .to_string(),
        strict_mode: false,
    }
}

fn test_guided_matcher_state(queue: &Queue) -> GuidedMatcherState {
    GuidedMatcherState::new(
        &test_guided_decoding_config(),
        &queue.guided_tokenizer_info,
        queue.guided_vocab_size,
    )
    .expect("guided matcher state")
}

fn test_tokenizer() -> Arc<Tokenizer> {
    test_tokenizer_with_vocab("1 \"{\" 1\n2 \"}\" 1\n3 \"a\" 1\n4 \":\" 1\n5 \"1\" 1\n")
}

fn test_tokenizer_with_vocab(vocab: &str) -> Arc<Tokenizer> {
    let vocab_path = std::env::temp_dir().join(format!("rwkv-infer-{}.txt", Uuid::new_v4()));
    fs::write(&vocab_path, vocab).expect("write test vocab");

    Arc::new(Tokenizer::new(vocab_path.to_str().expect("vocab path")).expect("tokenizer"))
}

fn wait_for_detokenize(queue: &mut Queue, item_id: usize) {
    for _ in 0..100 {
        queue.drain_detokenize_results();
        if queue
            .items
            .get(&item_id)
            .is_none_or(|item| item.pending_detokenize_tasks == 0)
        {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out waiting for detokenize");
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_string(),
            Err(_) => "non-string panic payload".to_string(),
        },
    }
}

fn build_guided_token_mask(vocab_size: usize, allowed_token_ids: &[i32]) -> Box<[i32]> {
    let mut guided_token_mask = vec![0_i32; vocab_size.div_ceil(32)];
    for &token_id in allowed_token_ids {
        let token_id = usize::try_from(token_id).expect("token id must be non-negative");
        let word_index = token_id / 32;
        let bit_index = token_id % 32;
        guided_token_mask[word_index] |= 1_i32 << bit_index;
    }
    guided_token_mask.into_boxed_slice()
}

fn bitmask_first_allowed_token_id(guided_token_mask: &[i32]) -> Option<i32> {
    for (word_index, &word) in guided_token_mask.iter().enumerate() {
        if word == 0 {
            continue;
        }

        for bit_index in 0..32 {
            let token_id = word_index * 32 + bit_index;
            if word & (1 << bit_index) != 0 {
                return Some(token_id as i32);
            }
        }
    }

    None
}
