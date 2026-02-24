#![allow(dead_code)]
use std::fmt;

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{Distribution, Int},
};
use rwkv_nn::kernels::wkv7_common::Wkv7ForwardInput;

#[cfg(feature = "cuda")]
pub type BenchBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(not(feature = "cuda"), feature = "rocm"))]
pub type BenchBackend = burn::backend::Rocm<f32, i32>;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
pub type BenchBackend = burn::backend::Vulkan<f32, i32>;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    feature = "metal"
))]
pub type BenchBackend = burn::backend::Metal<f32, i32>;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    not(feature = "metal")
))]
pub type BenchBackend = burn::backend::Wgpu<f32, i32>;

pub fn selected_backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        return "cuda";
    }

    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    {
        return "rocm";
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    {
        return "vulkan";
    }

    #[cfg(all(
        not(feature = "cuda"),
        not(feature = "rocm"),
        not(feature = "vulkan"),
        feature = "metal"
    ))]
    {
        return "metal";
    }

    "wgpu"
}

pub fn bench_device() -> <BenchBackend as Backend>::Device {
    <BenchBackend as Backend>::Device::default()
}

pub fn announce_backend() {
    println!("rwkv-nn kernel bench backend: {}", selected_backend_name());
}

#[derive(Clone, Copy, Debug)]
pub struct Wkv7Case {
    pub name: &'static str,
    pub batch_size: usize,
    pub context_len: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub chunk_len: usize,
}

impl Wkv7Case {
    pub const fn new(
        name: &'static str,
        batch_size: usize,
        context_len: usize,
        num_heads: usize,
        head_size: usize,
        chunk_len: usize,
    ) -> Self {
        Self {
            name,
            batch_size,
            context_len,
            num_heads,
            head_size,
            chunk_len,
        }
    }

    pub fn num_chunks(&self) -> usize {
        self.context_len / self.chunk_len
    }
}

impl fmt::Display for Wkv7Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.fmt(f)
    }
}

pub const WKV7_CASES: &[Wkv7Case] = &[
    Wkv7Case::new("b1_t128_h8_d64_c32", 1, 128, 8, 64, 32),
    Wkv7Case::new("b2_t256_h8_d64_c32", 2, 256, 8, 64, 32),
    Wkv7Case::new("b4_t512_h16_d64_c64", 4, 512, 16, 64, 64),
];

pub fn random_wkv7_input<B: Backend>(case: &Wkv7Case, device: &B::Device) -> Wkv7ForwardInput<B> {
    let shape = [
        case.batch_size,
        case.context_len,
        case.num_heads,
        case.head_size,
    ];

    Wkv7ForwardInput {
        weight_decay: Tensor::random(shape, Distribution::Normal(0.0, 1.0), device),
        receptance: Tensor::random(shape, Distribution::Normal(0.0, 1.0), device),
        replacement_key: Tensor::random(shape, Distribution::Normal(0.0, 1.0), device),
        value: Tensor::random(shape, Distribution::Normal(0.0, 1.0), device),
        removal_key_normalized: Tensor::random(shape, Distribution::Normal(0.0, 1.0), device),
        replacement: Tensor::random(shape, Distribution::Normal(0.0, 1.0), device),
    }
}

pub fn random_initial_state<B: Backend>(case: &Wkv7Case, device: &B::Device) -> Tensor<B, 4> {
    Tensor::random(
        [
            case.batch_size,
            case.num_heads,
            case.head_size,
            case.head_size,
        ],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

pub fn random_context_mask<B: Backend>(case: &Wkv7Case, device: &B::Device) -> Tensor<B, 2> {
    Tensor::random(
        [case.batch_size, case.context_len],
        Distribution::Bernoulli(0.95),
        device,
    )
}

pub fn random_output_grad<B: Backend>(case: &Wkv7Case, device: &B::Device) -> Tensor<B, 4> {
    Tensor::random(
        [
            case.batch_size,
            case.context_len,
            case.num_heads,
            case.head_size,
        ],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

pub fn random_final_state_grad<B: Backend>(case: &Wkv7Case, device: &B::Device) -> Tensor<B, 4> {
    Tensor::random(
        [
            case.batch_size,
            case.num_heads,
            case.head_size,
            case.head_size,
        ],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct RapidSampleCase {
    pub name: &'static str,
    pub batch_size: usize,
    pub vocab_size: usize,
    pub top_k: i32,
    pub top_p: f32,
}

impl RapidSampleCase {
    pub const fn new(
        name: &'static str,
        batch_size: usize,
        vocab_size: usize,
        top_k: i32,
        top_p: f32,
    ) -> Self {
        Self {
            name,
            batch_size,
            vocab_size,
            top_k,
            top_p,
        }
    }
}

impl fmt::Display for RapidSampleCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.fmt(f)
    }
}

pub const RAPID_SAMPLE_CASES: &[RapidSampleCase] = &[
    RapidSampleCase::new("b4_v4096", 4, 4096, 50, 0.95),
    RapidSampleCase::new("b8_v8192", 8, 8192, 100, 0.9),
    RapidSampleCase::new("b16_v16384", 16, 16384, 200, 0.9),
];

pub fn random_logits<B: Backend>(case: &RapidSampleCase, device: &B::Device) -> Tensor<B, 2> {
    Tensor::random(
        [case.batch_size, case.vocab_size],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

pub fn seed_states<B: Backend>(case: &RapidSampleCase, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::arange(0..case.batch_size as i64, device)
}

pub fn random_penalties<B: Backend>(case: &RapidSampleCase, device: &B::Device) -> Tensor<B, 2> {
    Tensor::random(
        [case.batch_size, case.vocab_size],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}
