use std::hint::black_box;

use burn::tensor::{Int, Tensor};
use burn::prelude::Backend;
use divan::Bencher;
use rwkv_nn::kernels::rapid_sample::{normalize_topk_topp, rapid_sample};

#[path = "../../common/mod.rs"]
mod common;

type B = common::BenchBackend;

fn build_sampling_params<B2: Backend>(
    case: &common::RapidSampleCase,
    device: &B2::Device,
) -> (Tensor<B2, 1>, Tensor<B2, 1, Int>, Tensor<B2, 1>) {
    let (tk, tp) = normalize_topk_topp(case.vocab_size, case.top_k, case.top_p);
    let inv_temps = Tensor::<B2, 1>::ones([case.batch_size], device);
    let top_ks = Tensor::<B2, 1, Int>::full([case.batch_size], tk as i32, device);
    let top_ps = Tensor::<B2, 1>::full([case.batch_size], tp, device);
    (inv_temps, top_ks, top_ps)
}

#[divan::bench(args = common::RAPID_SAMPLE_CASES)]
fn bench_rapid_sample_forward(bencher: Bencher<'_, '_>, case: &common::RapidSampleCase) {
    let device = common::bench_device();
    let logits = common::random_logits::<B>(case, &device);
    let states = common::seed_states::<B>(case, &device);
    let (inv_temps, top_ks, top_ps) = build_sampling_params::<B>(case, &device);

    bencher.bench_local(|| {
        black_box(rapid_sample(
            logits.clone(),
            states.clone(),
            inv_temps.clone(),
            top_ks.clone(),
            top_ps.clone(),
            None,
        ))
    });
}

#[divan::bench(args = common::RAPID_SAMPLE_CASES)]
fn bench_rapid_sample_forward_with_penalty(
    bencher: Bencher<'_, '_>,
    case: &common::RapidSampleCase,
) {
    let device = common::bench_device();
    let logits = common::random_logits::<B>(case, &device);
    let states = common::seed_states::<B>(case, &device);
    let penalties = common::random_penalties::<B>(case, &device);
    let (inv_temps, top_ks, top_ps) = build_sampling_params::<B>(case, &device);
    let pp = Tensor::<B, 1>::full([case.batch_size], 0.1f32, &device);
    let rp = Tensor::<B, 1>::full([case.batch_size], 0.2f32, &device);
    let pd = Tensor::<B, 1>::full([case.batch_size], 0.996f32, &device);

    bencher.bench_local(|| {
        black_box(rapid_sample(
            logits.clone(),
            states.clone(),
            inv_temps.clone(),
            top_ks.clone(),
            top_ps.clone(),
            Some((penalties.clone(), pp.clone(), rp.clone(), pd.clone())),
        ))
    });
}

fn main() {
    common::announce_backend();
    divan::main();
}
