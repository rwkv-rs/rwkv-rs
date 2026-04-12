use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use burn::{Tensor, tensor::Int};
use rwkv_nn::kernels::wkv7_infer::wkv7_infer_forward;

#[path = "../mod.rs"]
mod common;

type B = common::BenchBackend;

fn bench_wkv7_infer_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv-nn/kernels/wkv7_infer/forward");

    for case in common::WKV7_CASES {
        let device = common::bench_device();
        let input = common::random_wkv7_input::<B>(case, &device);
        let initial_state = common::random_initial_state::<B>(case, &device);
        let context_mask = common::random_context_mask::<B>(case, &device);
        let batch_ids = Tensor::<B, 1, Int>::arange(0..case.batch_size as i64, &device);
        let elapsed_t = Tensor::<B, 1, Int>::zeros([case.batch_size], &device);

        group.bench_with_input(BenchmarkId::from_parameter(case), case, |b, _case| {
            b.iter(|| {
                black_box(wkv7_infer_forward(
                    input.weight_decay.clone(),
                    input.receptance.clone(),
                    input.replacement_key.clone(),
                    input.value.clone(),
                    input.removal_key_normalized.clone(),
                    input.replacement.clone(),
                    batch_ids.clone(),
                    initial_state.clone(),
                    context_mask.clone(),
                    elapsed_t.clone(),
                ))
            });
        });
    }

    group.finish();
}

fn main() {
    common::announce_backend();
    let mut criterion = Criterion::default();
    bench_wkv7_infer_forward(&mut criterion);
    criterion.final_summary();
}
