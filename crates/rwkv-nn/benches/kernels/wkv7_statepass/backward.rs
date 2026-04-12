use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rwkv_nn::kernels::wkv7_statepass::{wkv7_statepass_backward, wkv7_statepass_forward};

#[path = "../mod.rs"]
mod common;

type B = common::BenchBackend;

fn bench_wkv7_statepass_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv-nn/kernels/wkv7_statepass/backward");

    for case in common::WKV7_CASES {
        let device = common::bench_device();
        let input = common::random_wkv7_input::<B>(case, &device);
        let initial_state = common::random_initial_state::<B>(case, &device);
        let forward = wkv7_statepass_forward(
            input.weight_decay.clone(),
            input.receptance.clone(),
            input.replacement_key.clone(),
            input.value.clone(),
            input.removal_key_normalized.clone(),
            input.replacement.clone(),
            initial_state,
            case.chunk_len,
        );
        let output_grad = common::random_output_grad::<B>(case, &device);
        let final_state_grad = common::random_final_state_grad::<B>(case, &device);

        group.bench_with_input(BenchmarkId::from_parameter(case), case, |b, _case| {
            b.iter(|| {
                black_box(wkv7_statepass_backward(
                    input.weight_decay.clone(),
                    input.receptance.clone(),
                    input.replacement_key.clone(),
                    input.value.clone(),
                    input.removal_key_normalized.clone(),
                    input.replacement.clone(),
                    forward.state.clone(),
                    forward.removal_state.clone(),
                    output_grad.clone(),
                    final_state_grad.clone(),
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
    bench_wkv7_statepass_backward(&mut criterion);
    criterion.final_summary();
}
