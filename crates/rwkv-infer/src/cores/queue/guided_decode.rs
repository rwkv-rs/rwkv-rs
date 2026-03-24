use std::thread;

use sonic_rs::{json, to_string};
use tokio::sync::mpsc;
use xgrammar::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor, GrammarCompiler, GrammarMatcher,
    TokenizerInfo, get_bitmask_shape, reset_token_bitmask,
};

use super::Queue;

pub(super) type GuidedDecodeTask = (usize, GuidedDecodingState);
pub(super) type GuidedDecodeResult = (
    usize,
    Result<(GuidedDecodingState, Option<Box<[i32]>>), String>,
);

pub(super) fn spawn_guided_decode_worker() -> (
    mpsc::UnboundedSender<GuidedDecodeTask>,
    mpsc::UnboundedReceiver<GuidedDecodeResult>,
) {
    let (guided_decode_sender, mut guided_decode_task_receiver) =
        mpsc::unbounded_channel::<GuidedDecodeTask>();
    let (guided_decode_result_sender, guided_decode_receiver) =
        mpsc::unbounded_channel::<GuidedDecodeResult>();

    thread::spawn(move || {
        while let Some((item_id, mut guided_decoding_state)) =
            guided_decode_task_receiver.blocking_recv()
        {
            let result = guided_decoding_state
                .fill_token_mask()
                .map(|guided_token_mask| (guided_decoding_state, guided_token_mask));

            if guided_decode_result_sender.send((item_id, result)).is_err() {
                break;
            }
        }
    });

    (guided_decode_sender, guided_decode_receiver)
}

impl Queue {
    pub(super) fn prepare_item_for_push(
        &mut self,
        item_id: usize,
        item: &mut super::QueueItem,
    ) -> Result<(), String> {
        let Some(guided_decoding_config) = item.guided_decoding_config.take() else {
            return Ok(());
        };

        let guided_decoding_state = GuidedDecodingState::new(
            &guided_decoding_config,
            &self.guided_tokenizer_info,
            self.guided_vocab_size,
        )?;

        self.guided_decode_sender
            .send((item_id, guided_decoding_state))
            .map_err(|_| "guided decode worker closed".to_string())?;
        item.guided_decoding_pending = true;
        Ok(())
    }

    pub(super) fn apply_guided_token_after_sample(
        &mut self,
        item_id: usize,
        token_id: i32,
        should_finish: &mut bool,
    ) -> Result<(), String> {
        let mut guided_decoding_state = {
            let item = self
                .items
                .get_mut(&item_id)
                .expect("scheduled item_id must exist in queue");
            let Some(guided_decoding_state) = item.guided_decoding_state.take() else {
                return Ok(());
            };

            item.guided_token_mask = None;
            guided_decoding_state
        };

        *should_finish |= guided_decoding_state.accept_token(token_id)?;

        if *should_finish {
            let item = self
                .items
                .get_mut(&item_id)
                .expect("scheduled item_id must exist in queue");
            item.guided_decoding_state = Some(guided_decoding_state);
            item.guided_decoding_pending = false;
            return Ok(());
        }

        self.guided_decode_sender
            .send((item_id, guided_decoding_state))
            .map_err(|_| "guided decode worker closed".to_string())?;

        let item = self
            .items
            .get_mut(&item_id)
            .expect("scheduled item_id must exist in queue");
        item.guided_decoding_pending = true;
        Ok(())
    }

    pub(super) fn drain_guided_decode_results(&mut self) {
        let mut removed_item_ids = Vec::new();

        while let Ok((item_id, result)) = self.guided_decode_receiver.try_recv() {
            let Some(item) = self.items.get_mut(&item_id) else {
                continue;
            };

            item.guided_decoding_pending = false;

            match result {
                Ok((guided_decoding_state, guided_token_mask)) => {
                    item.guided_decoding_state = Some(guided_decoding_state);
                    item.guided_token_mask = guided_token_mask;
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

pub struct GuidedDecodingState {
    matcher: GrammarMatcher,
    token_bitmask: Box<[i32]>,
    bitmask_shape: Vec<i64>,
    bitmask_strides: Vec<i64>,
}

impl std::fmt::Debug for GuidedDecodingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuidedDecodingState")
            .field("token_bitmask_len", &self.token_bitmask.len())
            .field("bitmask_shape", &self.bitmask_shape)
            .field("bitmask_strides", &self.bitmask_strides)
            .finish()
    }
}

// Safety: the matcher is always moved between the queue thread and a single
// background worker, and is never accessed concurrently.
unsafe impl Send for GuidedDecodingState {}

impl GuidedDecodingState {
    pub fn new(
        guided_decoding_config: &GuidedDecodingConfig,
        tokenizer_info: &TokenizerInfo,
        vocab_size: usize,
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
        let token_bitmask = xgrammar::allocate_token_bitmask(1, vocab_size);
        let (_, bitmask_size) = get_bitmask_shape(1, vocab_size);

        Ok(Self {
            matcher,
            token_bitmask,
            bitmask_shape: vec![1, bitmask_size as i64],
            bitmask_strides: vec![bitmask_size as i64, 1],
        })
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

    pub fn fill_token_mask(&mut self) -> Result<Option<Box<[i32]>>, String> {
        if self.matcher.is_terminated() {
            return Err("guided decoding matcher terminated before fill_token_mask".to_string());
        }

        reset_token_bitmask(&mut self.token_bitmask);
        let mut token_bitmask = DLTensor {
            data: self.token_bitmask.as_mut_ptr() as *mut std::ffi::c_void,
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
        };

        Ok(self
            .matcher
            .fill_next_token_bitmask(&mut token_bitmask, 0, false)
            .then(|| self.token_bitmask.clone()))
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
