use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rwkv_nn::kernels::wkv7_statepass::wkv7_statepass_forward;

#[path = "../mod.rs"]
mod common;

type B = common::BenchBackend;

fn bench_wkv7_statepass_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv-nn/kernels/wkv7_statepass/forward");

    for case in common::WKV7_CASES {
        let device = common::bench_device();
        let input = common::random_wkv7_input::<B>(case, &device);
        let initial_state = common::random_initial_state::<B>(case, &device);

        group.bench_with_input(BenchmarkId::from_parameter(case), case, |b, _case| {
            b.iter(|| {
                black_box(wkv7_statepass_forward(
                    input.weight_decay.clone(),
                    input.receptance.clone(),
                    input.replacement_key.clone(),
                    input.value.clone(),
                    input.removal_key_normalized.clone(),
                    input.replacement.clone(),
                    initial_state.clone(),
                    case.chunk_len,
                ))
            });
        });
    }

    group.finish();
}

fn main() {
    common::announce_backend();
    let mut criterion = Criterion::default();
    bench_wkv7_statepass_forward(&mut criterion);
    criterion.final_summary();
}
