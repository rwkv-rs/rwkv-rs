use burn::prelude::{Backend, Tensor};

use crate::kernels::wkv7_common::Wkv7ForwardInput;
use crate::kernels::wkv7_pretrain::{wkv7_pretrain_forward, Wkv7PretrainBackend};
use crate::kernels::wkv7_statepass::{wkv7_statepass_forward, Wkv7StatePassBackend};
use crate::kernels::wkv7_statetune::{wkv7_statetune_forward, Wkv7StateTuneBackend};

pub struct KernelOutput<B: Backend> {
    pub output: Tensor<B, 4>,
    pub next_state: Tensor<B, 4>,
}

pub trait Wkv7Kernel<B: Backend> {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Tensor<B, 4>,
        chunk_len: usize,
    ) -> KernelOutput<B>;
}

pub struct KernelPretrain;

impl<B: Wkv7PretrainBackend> Wkv7Kernel<B> for KernelPretrain {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Tensor<B, 4>,
        chunk_len: usize,
    ) -> KernelOutput<B> {
        let output = wkv7_pretrain_forward(input, chunk_len);
        KernelOutput {
            output: output.output,
            next_state: state,
        }
    }
}

pub struct KernelStatePass;

impl<B: Wkv7StatePassBackend> Wkv7Kernel<B> for KernelStatePass {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Tensor<B, 4>,
        chunk_len: usize,
    ) -> KernelOutput<B> {
        let output = wkv7_statepass_forward(
            input.weight_decay,
            input.receptance,
            input.replacement_key,
            input.value,
            input.removal_key_normalized,
            input.replacement,
            state,
            chunk_len,
        );

        KernelOutput {
            output: output.output,
            next_state: output.final_state,
        }
    }
}

pub struct KernelStateTune;

impl<B: Wkv7StateTuneBackend> Wkv7Kernel<B> for KernelStateTune {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Tensor<B, 4>,
        chunk_len: usize,
    ) -> KernelOutput<B> {
        let output = wkv7_statetune_forward(
            input.weight_decay,
            input.receptance,
            input.replacement_key,
            input.value,
            input.removal_key_normalized,
            input.replacement,
            state,
            chunk_len,
        );

        KernelOutput {
            output: output.output,
            next_state: output.state,
        }
    }
}
