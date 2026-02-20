use std::hint::black_box;

use divan::Bencher;
use rwkv_nn::kernels::rapid_sample::rapid_sample;

#[path = "../../common/mod.rs"]
mod common;

type B = common::BenchBackend;

#[divan::bench(args = common::RAPID_SAMPLE_CASES)]
fn bench_rapid_sample_forward(bencher: Bencher<'_, '_>, case: &common::RapidSampleCase) {
    let device = common::bench_device();
    let logits = common::random_logits::<B>(case, &device);
    let states = common::seed_states::<B>(case, &device);

    bencher.bench_local(|| {
        black_box(rapid_sample(
            logits.clone(),
            states.clone(),
            1.0,
            case.top_k,
            case.top_p,
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
    let penalty_cfg = common::default_penalty_config();

    bencher.bench_local(|| {
        black_box(rapid_sample(
            logits.clone(),
            states.clone(),
            1.0,
            case.top_k,
            case.top_p,
            Some((penalties.clone(), penalty_cfg)),
        ))
    });
}

fn main() {
    common::announce_backend();
    divan::main();
}
