use burn::cubecl;
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn apply_guided_token_masks_kernel(
    logits: &Tensor<f32>,
    batch_ids: &Tensor<u32>,
    guided_token_masks: &Tensor<u32>,
    masked_logits: &mut Tensor<f32>,
    vocab_size: u32,
    guided_token_mask_words: u32,
    guided_masked_logit: f32,
) {
    let index = ABSOLUTE_POS;
    if index >= masked_logits.len() {
        terminate!();
    }

    let vocab_size_usize = vocab_size as usize;
    let active_batch_index = index / vocab_size_usize;
    let token_id = index % vocab_size_usize;
    let word_index = token_id / 32;
    let bit_index = (token_id % 32) as u32;
    let batch_index = batch_ids[active_batch_index] as usize;
    let mask_index = batch_index * guided_token_mask_words as usize + word_index;
    let mask_word = guided_token_masks[mask_index];
    let allowed = (mask_word & (1u32 << bit_index)) != 0u32;

    masked_logits[index] = if allowed {
        logits[index]
    } else {
        guided_masked_logit
    };
}
