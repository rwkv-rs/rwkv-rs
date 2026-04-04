use std::{ffi::c_void, thread};

use sonic_rs::{json, to_string};
use tokio::sync::mpsc;
use xgrammar::{
    BatchGrammarMatcher,
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLTensor,
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    get_bitmask_shape,
    reset_token_bitmask,
};

use super::Queue;

pub(super) type GuidedDecodeTask = (usize, GuidedMatcherState);
pub(super) type GuidedDecodeResult = (usize, Result<GuidedPreparedState, String>);

#[derive(Debug)]
pub(super) enum GuidedDecodingStatus {
    Disabled,
    Pending,
    Ready(GuidedPreparedState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GuidedTokenMaskStatus {
    Masked,
    AllAllowed,
}

#[derive(Debug)]
pub(super) struct GuidedPreparedState {
    pub(super) matcher_state: GuidedMatcherState,
    pub(super) token_mask_status: GuidedTokenMaskStatus,
    token_mask_data: Option<Box<[i32]>>,
    token_mask_synced: bool,
}

impl GuidedPreparedState {
    pub(super) fn masked(matcher_state: GuidedMatcherState, token_mask_data: Box<[i32]>) -> Self {
        Self {
            matcher_state,
            token_mask_status: GuidedTokenMaskStatus::Masked,
            token_mask_data: Some(token_mask_data),
            token_mask_synced: false,
        }
    }

    pub(super) fn all_allowed(matcher_state: GuidedMatcherState) -> Self {
        Self {
            matcher_state,
            token_mask_status: GuidedTokenMaskStatus::AllAllowed,
            token_mask_data: None,
            token_mask_synced: false,
        }
    }

    pub(super) fn has_masked_token(&self) -> bool {
        self.token_mask_status == GuidedTokenMaskStatus::Masked
    }

    pub(super) fn take_token_mask_data(&mut self) -> Option<Box<[i32]>> {
        self.token_mask_data.take()
    }

    fn is_token_mask_synced(&self) -> bool {
        self.token_mask_synced
    }

    fn mark_token_mask_synced(&mut self) {
        self.token_mask_synced = true;
    }

    pub(super) fn allows_token(
        &self,
        token_mask_state: &GuidedTokenMaskBatchState,
        batch_id: usize,
        token_id: i32,
    ) -> bool {
        match self.token_mask_status {
            GuidedTokenMaskStatus::Masked => self
                .token_mask_data
                .as_deref()
                .map(|token_mask_data| bitmask_allows_token(token_mask_data, token_id))
                .unwrap_or_else(|| token_mask_state.allows_token(batch_id, token_id)),
            GuidedTokenMaskStatus::AllAllowed => token_id >= 0,
        }
    }
}

#[derive(Debug)]
pub(super) struct GuidedTokenMaskBatchState {
    token_masks: Box<[i32]>,
    token_mask_words: usize,
}

impl GuidedTokenMaskBatchState {
    pub(super) fn new(max_batch_size: usize, vocab_size: usize) -> Self {
        let token_masks = xgrammar::allocate_token_bitmask(max_batch_size, vocab_size);
        let (_, token_mask_words) = get_bitmask_shape(max_batch_size, vocab_size);

        Self {
            token_masks,
            token_mask_words,
        }
    }

    pub(super) fn reset(&mut self, batch_id: usize) {
        let row_range = self.row_range(batch_id);
        self.token_masks[row_range].fill(-1_i32);
    }

    pub(super) fn write_mask(&mut self, batch_id: usize, token_mask: &[i32]) {
        assert_eq!(
            token_mask.len(),
            self.token_mask_words,
            "guided token mask row has {} words, expected {}",
            token_mask.len(),
            self.token_mask_words
        );

        let row_range = self.row_range(batch_id);
        self.token_masks[row_range].copy_from_slice(token_mask);
    }

    pub(super) fn allows_token(&self, batch_id: usize, token_id: i32) -> bool {
        bitmask_allows_token(self.row(batch_id), token_id)
    }

    fn row(&self, batch_id: usize) -> &[i32] {
        let row_range = self.row_range(batch_id);
        &self.token_masks[row_range]
    }

    fn row_range(&self, batch_id: usize) -> std::ops::Range<usize> {
        let row_start = batch_id
            .checked_mul(self.token_mask_words)
            .expect("guided token mask row offset overflow");
        let row_end = row_start + self.token_mask_words;
        assert!(
            row_end <= self.token_masks.len(),
            "guided token mask batch_id {batch_id} out of range"
        );
        row_start..row_end
    }
}

pub(super) fn spawn_guided_decode_worker(
    vocab_size: usize,
) -> (
    mpsc::UnboundedSender<GuidedDecodeTask>,
    mpsc::UnboundedReceiver<GuidedDecodeResult>,
) {
    let (guided_decode_sender, mut guided_decode_task_receiver) =
        mpsc::unbounded_channel::<GuidedDecodeTask>();
    let (guided_decode_result_sender, guided_decode_receiver) =
        mpsc::unbounded_channel::<GuidedDecodeResult>();

    thread::spawn(move || {
        let mut worker = GuidedDecodeWorker::new(vocab_size);

        while let Some(first_task) = guided_decode_task_receiver.blocking_recv() {
            let mut tasks = vec![first_task];
            while let Ok(task) = guided_decode_task_receiver.try_recv() {
                tasks.push(task);
            }

            for result in worker.prepare_batch(tasks) {
                if guided_decode_result_sender.send(result).is_err() {
                    return;
                }
            }
        }
    });

    (guided_decode_sender, guided_decode_receiver)
}

struct GuidedDecodeWorker {
    batch_matcher: Option<BatchGrammarMatcher>,
    token_masks: Box<[i32]>,
    token_mask_words: usize,
    bitmask_shape: [i64; 2],
    bitmask_strides: [i64; 2],
}

impl GuidedDecodeWorker {
    fn new(vocab_size: usize) -> Self {
        let (_, token_mask_words) = get_bitmask_shape(1, vocab_size);

        Self {
            batch_matcher: BatchGrammarMatcher::new_auto().ok(),
            token_masks: Vec::new().into_boxed_slice(),
            token_mask_words,
            bitmask_shape: [0, token_mask_words as i64],
            bitmask_strides: [token_mask_words as i64, 1],
        }
    }

    fn prepare_batch(&mut self, tasks: Vec<GuidedDecodeTask>) -> Vec<GuidedDecodeResult> {
        if self.batch_matcher.is_some() {
            self.prepare_batch_parallel(tasks)
        } else {
            tasks
                .into_iter()
                .map(|(item_id, matcher_state)| {
                    (
                        item_id,
                        matcher_state.fill_token_mask(self.token_mask_words),
                    )
                })
                .collect()
        }
    }

    fn prepare_batch_parallel(&mut self, tasks: Vec<GuidedDecodeTask>) -> Vec<GuidedDecodeResult> {
        let mut results = Vec::with_capacity(tasks.len());
        let mut item_ids = Vec::with_capacity(tasks.len());
        let mut matcher_states = Vec::with_capacity(tasks.len());

        for (item_id, matcher_state) in tasks {
            if matcher_state.is_terminated() {
                results.push((
                    item_id,
                    Err("guided decoding matcher terminated before fill_token_mask".to_string()),
                ));
            } else {
                item_ids.push(item_id);
                matcher_states.push(matcher_state);
            }
        }

        if matcher_states.is_empty() {
            return results;
        }

        self.ensure_token_mask_capacity(matcher_states.len());
        reset_token_bitmask(&mut self.token_masks);

        let mut token_masks = self.token_masks_tensor(matcher_states.len());
        self.batch_matcher
            .as_mut()
            .expect("batch matcher must exist")
            .batch_fill_next_token_bitmask(
                GuidedMatcherState::matchers_ref(&matcher_states),
                &mut token_masks,
                None,
                false,
            );

        for (row_index, (item_id, matcher_state)) in
            item_ids.into_iter().zip(matcher_states).enumerate()
        {
            let row_start = row_index * self.token_mask_words;
            let row_end = row_start + self.token_mask_words;
            let row = &self.token_masks[row_start..row_end];

            let prepared_state = if bitmask_is_all_allowed(row) {
                GuidedPreparedState::all_allowed(matcher_state)
            } else {
                GuidedPreparedState::masked(matcher_state, row.to_vec().into_boxed_slice())
            };
            results.push((item_id, Ok(prepared_state)));
        }

        results
    }

    fn ensure_token_mask_capacity(&mut self, batch_size: usize) {
        let total_words = batch_size
            .checked_mul(self.token_mask_words)
            .expect("guided token mask batch size overflow");
        if self.token_masks.len() != total_words {
            self.token_masks = vec![-1_i32; total_words].into_boxed_slice();
        }
        self.bitmask_shape = [batch_size as i64, self.token_mask_words as i64];
        self.bitmask_strides = [self.token_mask_words as i64, 1];
    }

    fn token_masks_tensor(&mut self, batch_size: usize) -> DLTensor {
        debug_assert_eq!(
            self.token_masks.len(),
            batch_size * self.token_mask_words,
            "guided token mask worker scratch size mismatch"
        );

        DLTensor {
            data: self.token_masks.as_mut_ptr().cast::<c_void>(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            },
            shape: self.bitmask_shape.as_mut_ptr(),
            strides: self.bitmask_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    }
}

impl Queue {
    pub(super) fn set_guided_token_mask_row(
        &mut self,
        batch_id: usize,
        token_mask: Option<&[i32]>,
    ) {
        if let Some(token_mask) = token_mask {
            self.guided_token_mask_state
                .write_mask(batch_id, token_mask);
        } else {
            self.guided_token_mask_state.reset(batch_id);
        }
        self.model_forward
            .set_guided_token_mask_row(batch_id, token_mask);
    }

    fn sync_guided_prepared_state(
        &mut self,
        batch_id: usize,
        prepared_state: &mut GuidedPreparedState,
    ) {
        if prepared_state.is_token_mask_synced() {
            return;
        }

        match prepared_state.token_mask_status {
            GuidedTokenMaskStatus::Masked => {
                if let Some(token_mask_data) = prepared_state.take_token_mask_data() {
                    self.set_guided_token_mask_row(batch_id, Some(token_mask_data.as_ref()));
                }
            }
            GuidedTokenMaskStatus::AllAllowed => self.set_guided_token_mask_row(batch_id, None),
        }

        prepared_state.mark_token_mask_synced();
    }

    pub(super) fn prepare_item_for_push(
        &mut self,
        item_id: usize,
        item: &mut super::QueueItem,
    ) -> Result<(), String> {
        let Some(guided_decoding_config) = item.guided_decoding_config.take() else {
            return Ok(());
        };

        let guided_matcher_state = GuidedMatcherState::new(
            &guided_decoding_config,
            &self.guided_tokenizer_info,
            self.guided_vocab_size,
        )?;

        self.guided_decode_sender
            .send((item_id, guided_matcher_state))
            .map_err(|_| "guided decode worker closed".to_string())?;
        item.guided_decoding_status = GuidedDecodingStatus::Pending;
        Ok(())
    }

    pub(super) fn materialize_guided_prepared_states(&mut self, item_ids: &[usize]) {
        for &item_id in item_ids {
            let batch_id = self
                .items
                .get(&item_id)
                .expect("scheduled item_id must exist in queue")
                .batch_id
                .expect("batch_id should be assigned before guided materialization");
            let mut prepared_state = {
                let item = self
                    .items
                    .get_mut(&item_id)
                    .expect("scheduled item_id must exist in queue");
                match std::mem::replace(
                    &mut item.guided_decoding_status,
                    GuidedDecodingStatus::Disabled,
                ) {
                    GuidedDecodingStatus::Ready(prepared_state) => Some(prepared_state),
                    other_status => {
                        item.guided_decoding_status = other_status;
                        None
                    }
                }
            };

            if let Some(ref mut prepared_state) = prepared_state {
                self.sync_guided_prepared_state(batch_id, prepared_state);
            }

            if let Some(prepared_state) = prepared_state {
                self.items
                    .get_mut(&item_id)
                    .expect("scheduled item_id must exist in queue")
                    .guided_decoding_status = GuidedDecodingStatus::Ready(prepared_state);
            }
        }
    }

    pub(super) fn apply_guided_prepared_state(
        &mut self,
        item_id: usize,
        token_id: i32,
        requeue_next_step: bool,
    ) -> Result<bool, String> {
        let batch_id = self
            .items
            .get(&item_id)
            .expect("scheduled item_id must exist in queue")
            .batch_id
            .expect("scheduled item must have batch_id");
        let Some(mut prepared_state) = self.take_guided_prepared_state(item_id) else {
            return Ok(false);
        };

        assert!(
            prepared_state.allows_token(&self.guided_token_mask_state, batch_id, token_id),
            "sampler violated guided mask for item {item_id}, token {token_id}"
        );

        let terminated = prepared_state
            .matcher_state
            .accept_token(token_id)
            .unwrap_or_else(|err| {
                panic!("guided mask/state mismatch for item {item_id}, token {token_id}: {err}")
            });

        if terminated {
            let item = self
                .items
                .get_mut(&item_id)
                .expect("scheduled item_id must exist in queue");
            item.guided_decoding_status = GuidedDecodingStatus::Disabled;
            return Ok(true);
        }

        if !requeue_next_step {
            let item = self
                .items
                .get_mut(&item_id)
                .expect("scheduled item_id must exist in queue");
            item.guided_decoding_status = GuidedDecodingStatus::Disabled;
            return Ok(false);
        }

        self.guided_decode_sender
            .send((item_id, prepared_state.matcher_state))
            .map_err(|_| "guided decode worker closed".to_string())?;

        let item = self
            .items
            .get_mut(&item_id)
            .expect("scheduled item_id must exist in queue");
        item.guided_decoding_status = GuidedDecodingStatus::Pending;
        Ok(false)
    }

    fn take_guided_prepared_state(&mut self, item_id: usize) -> Option<GuidedPreparedState> {
        let item = self
            .items
            .get_mut(&item_id)
            .expect("scheduled item_id must exist in queue");

        match std::mem::replace(
            &mut item.guided_decoding_status,
            GuidedDecodingStatus::Disabled,
        ) {
            GuidedDecodingStatus::Disabled => None,
            GuidedDecodingStatus::Pending => {
                panic!("guided decoding sampled before prepared state was ready for item {item_id}")
            }
            GuidedDecodingStatus::Ready(prepared_state) => Some(prepared_state),
        }
    }

    pub(super) fn drain_guided_decode_results(&mut self) {
        let mut removed_item_ids = Vec::new();

        while let Ok((item_id, result)) = self.guided_decode_receiver.try_recv() {
            let Some(batch_id) = self.items.get(&item_id).map(|item| item.batch_id) else {
                continue;
            };

            match result {
                Ok(mut prepared_state) => {
                    if let Some(batch_id) = batch_id {
                        self.sync_guided_prepared_state(batch_id, &mut prepared_state);
                    }

                    if let Some(item) = self.items.get_mut(&item_id) {
                        item.guided_decoding_status = GuidedDecodingStatus::Ready(prepared_state);
                    }
                }
                Err(_) => removed_item_ids.push(item_id),
            }
        }

        if !removed_item_ids.is_empty() {
            removed_item_ids.sort_unstable();
            removed_item_ids.dedup();
            self.remove(&removed_item_ids);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuidedDecodingConfig {
    pub schema_json: String,
    pub strict_mode: bool,
}

#[repr(transparent)]
pub struct GuidedMatcherState {
    matcher: GrammarMatcher,
}

impl std::fmt::Debug for GuidedMatcherState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuidedMatcherState").finish()
    }
}

// Safety: the matcher is always moved between the queue thread and a single
// background worker, and is never accessed concurrently.
unsafe impl Send for GuidedMatcherState {}

impl GuidedMatcherState {
    pub fn new(
        guided_decoding_config: &GuidedDecodingConfig,
        tokenizer_info: &TokenizerInfo,
        _vocab_size: usize,
    ) -> Result<Self, String> {
        let mut compiler =
            GrammarCompiler::new(tokenizer_info, 1, true, -1).map_err(|err| err.to_string())?;
        let compiled = compiler
            .compile_json_schema(
                &guided_decoding_config.schema_json,
                true,
                None,
                None::<(&str, &str)>,
                guided_decoding_config.strict_mode,
                None,
            )
            .map_err(|err| err.to_string())?;
        let matcher =
            GrammarMatcher::new(&compiled, None, true, -1).map_err(|err| err.to_string())?;

        Ok(Self { matcher })
    }

    pub fn accept_token(&mut self, token_id: i32) -> Result<bool, String> {
        if self.matcher.is_terminated() {
            return Err("guided decoding matcher terminated before accept_token".to_string());
        }
        if !self.matcher.accept_token(token_id) {
            return Err(format!("guided decoding rejected sampled token {token_id}"));
        }
        Ok(self.matcher.is_terminated())
    }

    fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }

    fn fill_token_mask(mut self, token_mask_words: usize) -> Result<GuidedPreparedState, String> {
        if self.matcher.is_terminated() {
            return Err("guided decoding matcher terminated before fill_token_mask".to_string());
        }

        let mut token_mask = vec![-1_i32; token_mask_words].into_boxed_slice();
        let mut bitmask_shape = [1_i64, token_mask_words as i64];
        let mut bitmask_strides = [token_mask_words as i64, 1_i64];
        let mut token_masks = DLTensor {
            data: token_mask.as_mut_ptr().cast::<c_void>(),
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            },
            shape: bitmask_shape.as_mut_ptr(),
            strides: bitmask_strides.as_mut_ptr(),
            byte_offset: 0,
        };

        reset_token_bitmask(&mut token_mask);
        Ok(
            if self
                .matcher
                .fill_next_token_bitmask(&mut token_masks, 0, false)
            {
                GuidedPreparedState::masked(self, token_mask)
            } else {
                GuidedPreparedState::all_allowed(self)
            },
        )
    }

    fn matchers_ref(states: &[GuidedMatcherState]) -> &[GrammarMatcher] {
        // Safety: GuidedMatcherState is `#[repr(transparent)]` over `GrammarMatcher`.
        unsafe {
            std::slice::from_raw_parts(states.as_ptr().cast::<GrammarMatcher>(), states.len())
        }
    }
}

pub fn build_tokenizer_info_from_vocab(
    vocab_tokens: &[Vec<u8>],
    vocab_size: usize,
    stop_token_ids: &[i32],
) -> TokenizerInfo {
    let metadata = json!({
        "vocab_type": 0,
        "vocab_size": vocab_size,
        "add_prefix_space": false,
        "stop_token_ids": stop_token_ids,
    });

    TokenizerInfo::from_vocab_and_metadata_bytes(
        vocab_tokens.iter().map(Vec::as_slice),
        &to_string(&metadata).unwrap_or_else(|_| "{}".to_string()),
    )
}

fn bitmask_allows_token(token_mask: &[i32], token_id: i32) -> bool {
    let Ok(token_id) = usize::try_from(token_id) else {
        return false;
    };
    let word_index = token_id / 32;
    let bit_index = token_id % 32;
    token_mask
        .get(word_index)
        .is_some_and(|&word| ((word as u32) & (1_u32 << bit_index)) != 0)
}

fn bitmask_is_all_allowed(token_mask: &[i32]) -> bool {
    token_mask.iter().all(|&word| word == -1_i32)
}
