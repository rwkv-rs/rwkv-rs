use crate::kernels::wkv7_infer::{Wkv7InferBackend, wkv7_infer_forward};
use crate::kernels::wkv7_pretrain::{Wkv7PretrainBackend, wkv7_pretrain_forward};
use crate::kernels::wkv7_statepass::{Wkv7StatePassBackend, wkv7_statepass_forward};
use crate::kernels::wkv7_statetune::{Wkv7StateTuneBackend, wkv7_statetune_forward};
use burn::Tensor;
use burn::prelude::Backend;

pub mod host;
pub mod kernel;

pub trait Wkv7Backend:
    Wkv7PretrainBackend + Wkv7StateTuneBackend + Wkv7StatePassBackend + Wkv7InferBackend
{
}
impl<B> Wkv7Backend for B where
    B: Wkv7PretrainBackend + Wkv7StateTuneBackend + Wkv7StatePassBackend + Wkv7InferBackend
{
}

pub trait Wkv7Kernel<B: Backend> {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Option<Tensor<B, 4>>,
        chunk_len: usize,
    ) -> KernelOutput<B>;
}

pub struct KernelPretrain;

impl<B: Wkv7PretrainBackend> Wkv7Kernel<B> for KernelPretrain {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Option<Tensor<B, 4>>,
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
        state: Option<Tensor<B, 4>>,
        chunk_len: usize,
    ) -> KernelOutput<B> {
        let output = wkv7_statepass_forward(
            input.weight_decay,
            input.receptance,
            input.replacement_key,
            input.value,
            input.removal_key_normalized,
            input.replacement,
            state.unwrap(),
            chunk_len,
        );

        KernelOutput {
            output: output.output,
            next_state: Some(output.final_state),
        }
    }
}

pub struct KernelStateTune;

impl<B: Wkv7StateTuneBackend> Wkv7Kernel<B> for KernelStateTune {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Option<Tensor<B, 4>>,
        chunk_len: usize,
    ) -> KernelOutput<B> {
        let output = wkv7_statetune_forward(
            input.weight_decay,
            input.receptance,
            input.replacement_key,
            input.value,
            input.removal_key_normalized,
            input.replacement,
            state.unwrap(),
            chunk_len,
        );

        KernelOutput {
            output: output.output,
            next_state: Some(output.state),
        }
    }
}

/// Inference-only WKV7 kernel.
///
/// Uses the lightweight WKV7 inference implementation that returns only
/// `output` and `final_state` (no chunk state snapshots, no backward).
pub struct KernelInfer;

impl<B: Wkv7InferBackend> Wkv7Kernel<B> for KernelInfer {
    fn forward(
        input: Wkv7ForwardInput<B>,
        state: Option<Tensor<B, 4>>,
        _chunk_len: usize,
    ) -> KernelOutput<B> {
        let initial_state = state.expect("initial_state required");
        let [batch_size, context_length, _num_heads, _head_size] = input.weight_decay.dims();
        let device = input.weight_decay.device();
        let context_mask = Tensor::ones([batch_size, context_length], &device);

        let output = wkv7_infer_forward(
            input.weight_decay,
            input.receptance,
            input.replacement_key,
            input.value,
            input.removal_key_normalized,
            input.replacement,
            initial_state,
            context_mask,
        );

        KernelOutput {
            output: output.output,
            next_state: Some(output.final_state),
        }
    }
}

pub struct KernelOutput<B: Backend> {
    pub output: Tensor<B, 4>,
    pub next_state: Option<Tensor<B, 4>>,
}

#[derive(Clone, Debug)]
pub struct Wkv7ForwardInput<B: Backend> {
    pub receptance: Tensor<B, 4>,
    pub weight_decay: Tensor<B, 4>,
    pub replacement_key: Tensor<B, 4>,
    pub value: Tensor<B, 4>,
    pub removal_key_normalized: Tensor<B, 4>,
    pub replacement: Tensor<B, 4>,
}

#[derive(Clone, Debug)]
pub struct Wkv7ForwardOutput<T> {
    pub state: T,
    pub removal_state: T,
    pub output: T,
}

#[derive(Clone, Debug)]
pub struct Wkv7StatePassForwardOutput<T> {
    pub state: T,
    pub removal_state: T,
    pub output: T,
    pub final_state: T,
}

#[derive(Clone, Debug)]
pub struct Wkv7BackwardOutput<T> {
    pub weight_decay_grad: T,
    pub receptance_grad: T,
    pub key_grad: T,
    pub value_grad: T,
    pub removal_grad: T,
    pub replacement_grad: T,
}

#[derive(Clone, Debug)]
pub struct Wkv7StateBackwardOutput<T> {
    pub weight_decay_grad: T,
    pub receptance_grad: T,
    pub key_grad: T,
    pub value_grad: T,
    pub removal_grad: T,
    pub replacement_grad: T,
    pub initial_state_grad: T,
}
