use std::{
    fs,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::{END_TOKEN_ID, Queue, QueueItem};
use crate::cores::forward::{Logits, ModelForward, TokenId, TokenIdLogprobsConfig};
use crate::cores::guided_decoding::GuidedDecodingConfig;

#[derive(Default)]
struct DummyModelForward {
    choose_first_allowed_token: bool,
    fixed_token_id: i32,
    sample_guided_token_masks: Arc<Mutex<Vec<Vec<Option<Vec<i32>>>>>>,
}

impl ModelForward for DummyModelForward {
    fn forward(
        &mut self,
        batch_ids: &[usize],
        _contexts: &[&[i32]],
        _context_masks: &[&[u8]],
    ) -> Vec<Logits> {
        batch_ids
            .iter()
            .map(|&batch_index| Logits {
                batch_index,
                logits: vec![0.0; 8],
            })
            .collect()
    }

    fn sample(
        &mut self,
        logits: Vec<Logits>,
        _sampling_configs: &[crate::cores::forward::sampling::SamplingConfig],
        _token_logprobs_configs: &[Option<TokenIdLogprobsConfig>],
        guided_token_masks: &[Option<&[i32]>],
    ) -> Vec<TokenId> {
        self.sample_guided_token_masks.lock().unwrap().push(
            guided_token_masks
                .iter()
                .map(|guided_token_mask| guided_token_mask.map(|mask| mask.to_vec()))
                .collect(),
        );

        logits
            .into_iter()
            .zip(guided_token_masks.iter())
            .map(|(logits, guided_token_mask)| {
                let token_id = if self.choose_first_allowed_token {
                    guided_token_mask
                        .and_then(bitmask_first_allowed_token_id)
                        .unwrap_or(END_TOKEN_ID)
                } else {
                    self.fixed_token_id
                };

                TokenId {
                    batch_index: logits.batch_index,
                    token_id,
                    logprob: None,
                    finish_after_token: false,
                }
            })
            .collect()
    }

    fn reset(&mut self, _batch_index: usize) {}
}

#[test]
fn prefill_without_output_does_not_wait_for_guided_token_mask() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        2,
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
fn prefill_waits_for_guided_token_mask() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        2,
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

        if queue
            .items
            .get(&1)
            .is_some_and(|item| item.guided_token_mask.is_some())
        {
            break;
        }

        thread::sleep(Duration::from_millis(1));
    }

    assert!(
        queue
            .items
            .get(&1)
            .is_some_and(|item| item.guided_token_mask.is_some())
    );
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
        }),
        test_tokenizer(),
        4,
        2,
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

    let item = queue.items.get_mut(&1).expect("queue item");
    item.guided_token_mask = Some(vec![123, 456].into_boxed_slice());

    queue.update_batch_status();
    let item_ids = queue.collect_step_item_ids();
    assert_eq!(item_ids, vec![1]);

    let new_tokens = queue.step(&item_ids).expect("sample output");
    assert_eq!(new_tokens.len(), 1);

    let sample_guided_token_masks = sample_guided_token_masks.lock().unwrap().clone();
    assert_eq!(sample_guided_token_masks, vec![vec![Some(vec![123, 456])]]);
}

#[test]
fn run_guided_item_end_to_end() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward {
            choose_first_allowed_token: true,
            fixed_token_id: END_TOKEN_ID,
            sample_guided_token_masks: Arc::new(Mutex::new(Vec::new())),
        }),
        test_tokenizer(),
        4,
        1,
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
    while let Ok(delta) = completions_rx.try_recv() {
        completions_text.push_str(&delta);
    }

    assert_eq!(completions_text, "{}");
    assert!(queue.items.is_empty());
}

#[test]
fn late_guided_result_is_ignored_after_remove() {
    let mut queue = Queue::new(
        Box::new(DummyModelForward::default()),
        test_tokenizer(),
        4,
        1,
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

fn test_guided_decoding_config() -> GuidedDecodingConfig {
    GuidedDecodingConfig {
        schema_json: r#"{"type":"object","properties":{},"additionalProperties":false}"#
            .to_string(),
        strict_mode: false,
    }
}

fn test_tokenizer() -> Arc<Tokenizer> {
    let vocab_path = std::env::temp_dir().join(format!("rwkv-infer-{}.txt", Uuid::new_v4()));
    fs::write(
        &vocab_path,
        "1 \"{\" 1\n2 \"}\" 1\n3 \"a\" 1\n4 \":\" 1\n5 \"1\" 1\n",
    )
    .expect("write test vocab");

    Arc::new(Tokenizer::new(vocab_path.to_str().expect("vocab path")).expect("tokenizer"))
}

fn bitmask_first_allowed_token_id(guided_token_mask: &[i32]) -> Option<i32> {
    for (word_index, &word) in guided_token_mask.iter().enumerate() {
        if word == 0 {
            continue;
        }

        for bit_index in 0..32 {
            let token_id = word_index * 32 + bit_index;
            if token_id == 0 {
                continue;
            }
            if word & (1 << bit_index) != 0 {
                return Some(token_id as i32);
            }
        }
    }

    None
}
