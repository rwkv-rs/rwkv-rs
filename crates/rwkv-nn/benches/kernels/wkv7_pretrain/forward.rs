use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rwkv_nn::kernels::wkv7_pretrain::wkv7_pretrain_forward;

#[path = "../mod.rs"]
mod common;

type B = common::BenchBackend;

fn bench_wkv7_pretrain_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv-nn/kernels/wkv7_pretrain/forward");

    for case in common::WKV7_CASES {
        let device = common::bench_device();
        let input = common::random_wkv7_input::<B>(case, &device);

        group.bench_with_input(BenchmarkId::from_parameter(case), case, |b, _case| {
            b.iter(|| black_box(wkv7_pretrain_forward(input.clone(), case.chunk_len)));
        });
    }

    group.finish();
}

fn main() {
    common::announce_backend();
    let mut criterion = Criterion::default();
    bench_wkv7_pretrain_forward(&mut criterion);
    criterion.final_summary();
}
