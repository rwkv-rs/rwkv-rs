use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;

use crate::cores::forward::sampling::SamplingConfig;
use crate::cores::forward::{ModelForward, TokenId, TokenIdLogprobsConfig};

pub struct Queue {
    model_forward: Box<dyn ModelForward>,
    max_batch_size: usize,
    paragraph_len: usize,
    items: Vec<QueueItem>,
    batch_status: BatchStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchStatus {
    PrefillWithoutOutput,
    Prefill,
    Decode,
}

impl Queue {
    pub fn new(
        model_forward: Box<dyn ModelForward>,
        max_batch_size: usize,
        paragraph_len: usize,
    ) -> Self {
        debug_assert!(max_batch_size > 0);
        debug_assert!(paragraph_len > 0);

        Self {
            model_forward,
            max_batch_size,
            paragraph_len,
            items: Vec::new(),
            batch_status: BatchStatus::PrefillWithoutOutput,
        }
    }

    pub fn push(&mut self, item: QueueItem) {
        // 约定: item 入队前已经完成 tokenize.
        // 1. 首部先补一个 0.
        // 2. 若总长度仍不能被 paragraph_len 整除, 继续在首部补 0 直到整除.
        // 3. 此时 paragraph_count = context_tokens_for_step.len() / paragraph_len.
        // 4. Waiting / Prefill 阶段, context_tokens_for_step 存完整 prompt.
        // 5. Decode 阶段, context_tokens_for_step 只存上一轮新 token.
        self.items.push(item);
        self.update_batch_status();
    }

    pub fn step(&mut self, item_ids: Vec<usize>) -> Option<Vec<TokenId>> {
        if item_ids.is_empty() {
            return None;
        }

        let item_indexes: Vec<usize> = item_ids
            .iter()
            .map(|item_id| {
                self.items
                    .iter()
                    .position(|item| item.item_id == *item_id)
                    .expect("scheduled item_id must exist in queue")
            })
            .collect();

        let mut used_batch_ids: HashSet<usize> =
            self.items.iter().filter_map(|item| item.batch_id).collect();

        for &item_index in &item_indexes {
            if self.items[item_index].batch_id.is_some() {
                continue;
            }

            let batch_id = (0..self.max_batch_size)
                .find(|candidate| !used_batch_ids.contains(candidate))
                .expect("scheduler should not select more items than free batch slots");

            self.items[item_index].batch_id = Some(batch_id);
            used_batch_ids.insert(batch_id);
        }

        let mut batch_ids = Vec::with_capacity(item_indexes.len());
        let mut contexts = Vec::with_capacity(item_indexes.len());
        let mut masks = Vec::with_capacity(item_indexes.len());

        for &item_index in &item_indexes {
            let item = &self.items[item_index];
            let batch_id = item.batch_id.expect("batch_id should be assigned before forward");
            batch_ids.push(batch_id);

            match self.batch_status {
                BatchStatus::PrefillWithoutOutput | BatchStatus::Prefill => {
                    let next_paragraph_id = match item.status {
                        QueueItemStatus::Waiting => 0,
                        QueueItemStatus::Prefill(next_paragraph_id) => next_paragraph_id,
                        QueueItemStatus::Decode(_) => {
                            panic!("decode item should not be scheduled in prefill")
                        }
                    };

                    let start = next_paragraph_id * self.paragraph_len;
                    let end = start + self.paragraph_len;
                    let context = item.context_tokens_for_step[start..end].to_vec();

                    let leading_zero_count =
                        context.iter().take_while(|&&token_id| token_id == 0).count();
                    let mut mask = vec![1u8; context.len()];
                    for index in 0..leading_zero_count.saturating_sub(1) {
                        mask[index] = 0;
                    }

                    contexts.push(context);
                    masks.push(mask);
                }
                BatchStatus::Decode => {
                    contexts.push(item.context_tokens_for_step.clone());
                    masks.push(vec![1u8; item.context_tokens_for_step.len()]);
                }
            }
        }

        let context_refs: Vec<&[i32]> = contexts.iter().map(Vec::as_slice).collect();
        let mask_refs: Vec<&[u8]> = masks.iter().map(Vec::as_slice).collect();
        let logits = self
            .model_forward
            .forward(&batch_ids, &context_refs, &mask_refs);

        if self.batch_status == BatchStatus::PrefillWithoutOutput {
            return None;
        }

        let sampling_configs: Vec<SamplingConfig> = item_indexes
            .iter()
            .map(|&item_index| self.items[item_index].sampling_config)
            .collect();
        let token_logprobs_configs: Vec<Option<TokenIdLogprobsConfig>> = item_indexes
            .iter()
            .map(|&item_index| self.items[item_index].token_logprobs_config.clone())
            .collect();

        Some(self.model_forward.sample(
            &batch_ids,
            logits,
            &sampling_configs,
            &token_logprobs_configs,
        ))
    }

    pub fn remove(&mut self, item_ids: &[usize]) {
        let item_id_set: HashSet<usize> = item_ids.iter().copied().collect();
        let mut item_indexes: Vec<usize> = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| item_id_set.contains(&item.item_id).then_some(index))
            .collect();

        item_indexes.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));

        for item_index in item_indexes {
            if let Some(batch_id) = self.items[item_index].batch_id.take() {
                self.model_forward.reset(batch_id);
            }
            self.items.remove(item_index);
        }

        self.update_batch_status();
    }

    pub fn run(&mut self) {
        while !self.items.is_empty() {
            let mut item_ids = Vec::with_capacity(self.max_batch_size);

            match self.batch_status {
                BatchStatus::PrefillWithoutOutput => {
                    for item in &self.items {
                        if item_ids.len() == self.max_batch_size {
                            break;
                        }
                        if matches!(
                            item.status,
                            QueueItemStatus::Prefill(next_paragraph_id)
                                if next_paragraph_id + 1 < item.paragraph_count
                        ) {
                            item_ids.push(item.item_id);
                        }
                    }

                    for item in &self.items {
                        if item_ids.len() == self.max_batch_size {
                            break;
                        }
                        if matches!(item.status, QueueItemStatus::Waiting)
                            && item.paragraph_count > 1
                        {
                            item_ids.push(item.item_id);
                        }
                    }

                    if item_ids.is_empty() {
                        self.update_batch_status();
                        continue;
                    }

                    let _ = self.step(item_ids.clone());

                    for item_id in item_ids {
                        let item = self
                            .items
                            .iter_mut()
                            .find(|item| item.item_id == item_id)
                            .expect("scheduled item_id must exist in queue");

                        item.status = match item.status {
                            QueueItemStatus::Waiting => QueueItemStatus::Prefill(1),
                            QueueItemStatus::Prefill(next_paragraph_id) => {
                                QueueItemStatus::Prefill(next_paragraph_id + 1)
                            }
                            QueueItemStatus::Decode(_) => {
                                panic!("decode item should not be scheduled in prefill_without_output")
                            }
                        };
                    }
                }
                BatchStatus::Prefill => {
                    for item in &self.items {
                        if item_ids.len() == self.max_batch_size {
                            break;
                        }
                        if matches!(
                            item.status,
                            QueueItemStatus::Prefill(next_paragraph_id)
                                if next_paragraph_id + 1 == item.paragraph_count
                        ) {
                            item_ids.push(item.item_id);
                        }
                    }

                    for item in &self.items {
                        if item_ids.len() == self.max_batch_size {
                            break;
                        }
                        if matches!(item.status, QueueItemStatus::Waiting)
                            && item.paragraph_count == 1
                        {
                            item_ids.push(item.item_id);
                        }
                    }

                    if item_ids.is_empty() {
                        self.update_batch_status();
                        continue;
                    }

                    let new_tokens = self.step(item_ids.clone()).unwrap();
                    self.handle_output_tokens(&item_ids, new_tokens, true);
                }
                BatchStatus::Decode => {
                    for item in &self.items {
                        if item_ids.len() == self.max_batch_size {
                            break;
                        }
                        if matches!(
                            item.status,
                            QueueItemStatus::Decode(new_tokens_len)
                                if new_tokens_len < item.sampling_config.max_new_tokens
                        ) {
                            item_ids.push(item.item_id);
                        }
                    }

                    if item_ids.is_empty() {
                        self.update_batch_status();
                        continue;
                    }

                    let new_tokens = self.step(item_ids.clone()).unwrap();
                    self.handle_output_tokens(&item_ids, new_tokens, false);
                }
            }

            self.update_batch_status();
        }
    }

    pub fn update_batch_status(&mut self) {
        if self.items.is_empty() {
            self.batch_status = BatchStatus::PrefillWithoutOutput;
            return;
        }

        let mut has_prefill_without_output = false;
        let mut has_prefill = false;
        let mut has_decode = false;
        let mut has_waiting_single_paragraph = false;
        let mut has_waiting_multi_paragraph = false;

        for item in &self.items {
            match item.status {
                QueueItemStatus::Waiting => {
                    if item.paragraph_count == 1 {
                        has_waiting_single_paragraph = true;
                    } else {
                        has_waiting_multi_paragraph = true;
                    }
                }
                QueueItemStatus::Prefill(next_paragraph_id) => {
                    if next_paragraph_id + 1 < item.paragraph_count {
                        has_prefill_without_output = true;
                    } else {
                        has_prefill = true;
                    }
                }
                QueueItemStatus::Decode(new_tokens_len) => {
                    if new_tokens_len < item.sampling_config.max_new_tokens {
                        has_decode = true;
                    }
                }
            }
        }

        self.batch_status = match self.batch_status {
            BatchStatus::PrefillWithoutOutput => {
                if has_decode {
                    BatchStatus::Decode
                } else if has_prefill || has_waiting_single_paragraph {
                    BatchStatus::Prefill
                } else {
                    BatchStatus::PrefillWithoutOutput
                }
            }
            BatchStatus::Prefill => {
                if has_decode {
                    BatchStatus::Decode
                } else if has_prefill_without_output || has_waiting_multi_paragraph {
                    BatchStatus::PrefillWithoutOutput
                } else if has_prefill || has_waiting_single_paragraph {
                    BatchStatus::Prefill
                } else {
                    BatchStatus::PrefillWithoutOutput
                }
            }
            BatchStatus::Decode => {
                if has_prefill || has_waiting_single_paragraph {
                    BatchStatus::Prefill
                } else if has_prefill_without_output || has_waiting_multi_paragraph {
                    BatchStatus::PrefillWithoutOutput
                } else if has_decode {
                    BatchStatus::Decode
                } else {
                    BatchStatus::PrefillWithoutOutput
                }
            }
        };
    }

    fn handle_output_tokens(
        &mut self,
        item_ids: &[usize],
        new_tokens: Vec<TokenId>,
        from_prefill: bool,
    ) {
        let token_by_batch_id: HashMap<usize, TokenId> = new_tokens
            .into_iter()
            .map(|token| (token.batch_index, token))
            .collect();

        let mut removed_item_ids = Vec::new();

        for &item_id in item_ids {
            let item = self
                .items
                .iter_mut()
                .find(|item| item.item_id == item_id)
                .expect("scheduled item_id must exist in queue");

            let batch_id = item.batch_id.expect("scheduled item must have batch_id");
            let Some(token) = token_by_batch_id.get(&batch_id) else {
                // TODO: sample 返回结果缺失时, 这里应该转成内部错误并移出相关 item.
                continue;
            };

            // TODO: tokenize / 文本输出相关逻辑仍保留伪代码:
            // 1. 异步 detokenize, 不阻塞当前 forward 线程.
            // 2. 若结果是不完整 UTF-8, 先累积到 detokenize_buffer.
            // 3. 若本轮刚好拼成完整 UTF-8, 得到 delta_text.
            // 4. 把 delta_text 追加到 completions_text.
            // 5. 检查 stop_suffixes:
            //    - 若命中, 删掉 stop_suffix 之后的文本
            //    - 并把该 item 标记为 remove
            // 6. completions_tx 只发送本轮新增的 delta_text.
            // 7. 若采到了 END token, 这里也应该把 item 标记为 remove.

            if token.finish_after_token {
                removed_item_ids.push(item_id);
                continue;
            }

            if from_prefill {
                if item.sampling_config.max_new_tokens <= 1 {
                    removed_item_ids.push(item_id);
                } else {
                    item.status = QueueItemStatus::Decode(1);
                    item.context_tokens_for_step = vec![token.token_id];
                }
                continue;
            }

            let next_len = match item.status {
                QueueItemStatus::Decode(old_len) => old_len + 1,
                QueueItemStatus::Waiting | QueueItemStatus::Prefill(_) => {
                    panic!("non-decode item should not be scheduled in decode")
                }
            };

            if next_len >= item.sampling_config.max_new_tokens {
                removed_item_ids.push(item_id);
            } else {
                item.status = QueueItemStatus::Decode(next_len);
                item.context_tokens_for_step = vec![token.token_id];
            }
        }

        if !removed_item_ids.is_empty() {
            self.remove(&removed_item_ids);
        }
    }
}

pub struct QueueItem {
    item_id: usize,
    batch_id: Option<usize>,
    paragraph_count: usize,
    context_tokens_for_step: Vec<i32>,

    sampling_config: SamplingConfig,
    token_logprobs_config: Option<TokenIdLogprobsConfig>,
    #[allow(dead_code)]
    stop_suffixes: Vec<String>,
    status: QueueItemStatus,

    #[allow(dead_code)]
    detokenize_buffer: Vec<i32>,
    #[allow(dead_code)]
    completions_text: String,
    #[allow(dead_code)]
    completions_tx: mpsc::Sender<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueItemStatus {
    Waiting,
    Prefill(usize), // next_paragraph_id, 0-based
    Decode(usize), // new_tokens_len
}
