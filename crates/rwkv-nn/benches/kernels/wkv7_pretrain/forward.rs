use std::hint::black_box;

use divan::Bencher;
use rwkv_nn::kernels::wkv7_pretrain::wkv7_pretrain_forward;

#[path = "../../common/mod.rs"]
mod common;

type B = common::BenchBackend;

#[divan::bench(args = common::WKV7_CASES)]
fn bench_wkv7_pretrain_forward(bencher: Bencher<'_, '_>, case: &common::Wkv7Case) {
    let device = common::bench_device();
    let input = common::random_wkv7_input::<B>(case, &device);

    bencher.bench_local(|| black_box(wkv7_pretrain_forward(input.clone(), case.chunk_len)));
}

fn main() {
    common::announce_backend();
    divan::main();
}
