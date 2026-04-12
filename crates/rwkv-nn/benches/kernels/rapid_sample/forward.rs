use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};
use rwkv_nn::kernels::rapid_sample::rapid_sample;

#[path = "../mod.rs"]
mod common;

type B = common::BenchBackend;

fn build_sampling_params<B2: Backend>(
    case: &common::RapidSampleCase,
    device: &B2::Device,
) -> (Tensor<B2, 1>, Tensor<B2, 1, Int>, Tensor<B2, 1>) {
    let (tk, tp) = (case.top_k, case.top_p);
    let inv_temps = Tensor::<B2, 1>::ones([case.batch_size], device);
    let top_ks = Tensor::<B2, 1, Int>::full([case.batch_size], tk, device);
    let top_ps = Tensor::<B2, 1>::full([case.batch_size], tp, device);
    (inv_temps, top_ks, top_ps)
}

fn bench_rapid_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv-nn/kernels/rapid_sample/forward");

    for case in common::RAPID_SAMPLE_CASES {
        let device = common::bench_device();
        let logits = common::random_logits::<B>(case, &device);
        let batch_ids = Tensor::<B, 1, Int>::arange(0..case.batch_size as i64, &device);
        let states = common::seed_states::<B>(case, &device);
        let penalties = common::random_penalties::<B>(case, &device);
        let (inv_temps, top_ks, top_ps) = build_sampling_params::<B>(case, &device);
        let pp = Tensor::<B, 1>::full([case.batch_size], 0.1f32, &device);
        let rp = Tensor::<B, 1>::full([case.batch_size], 0.2f32, &device);
        let pd = Tensor::<B, 1>::full([case.batch_size], 0.996f32, &device);

        group.bench_with_input(BenchmarkId::from_parameter(case), case, |b, _case| {
            b.iter(|| {
                black_box(rapid_sample(
                    logits.clone(),
                    batch_ids.clone(),
                    states.clone(),
                    inv_temps.clone(),
                    top_ks.clone(),
                    top_ps.clone(),
                    (penalties.clone(), pp.clone(), rp.clone(), pd.clone()),
                ))
            });
        });
    }

    group.finish();
}

fn main() {
    common::announce_backend();
    let mut criterion = Criterion::default();
    bench_rapid_sample(&mut criterion);
    criterion.final_summary();
}
