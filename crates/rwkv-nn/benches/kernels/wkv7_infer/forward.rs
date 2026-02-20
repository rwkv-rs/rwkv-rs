use std::hint::black_box;

use divan::Bencher;
use rwkv_nn::kernels::wkv7_infer::wkv7_infer_forward;

#[path = "../../common/mod.rs"]
mod common;

type B = common::BenchBackend;

#[divan::bench(args = common::WKV7_CASES)]
fn bench_wkv7_infer_forward(bencher: Bencher<'_, '_>, case: &common::Wkv7Case) {
    let device = common::bench_device();
    let input = common::random_wkv7_input::<B>(case, &device);
    let initial_state = common::random_initial_state::<B>(case, &device);
    let context_mask = common::random_context_mask::<B>(case, &device);

    bencher.bench_local(|| {
        black_box(wkv7_infer_forward(
            input.weight_decay.clone(),
            input.receptance.clone(),
            input.replacement_key.clone(),
            input.value.clone(),
            input.removal_key_normalized.clone(),
            input.replacement.clone(),
            initial_state.clone(),
            context_mask.clone(),
        ))
    });
}

fn main() {
    common::announce_backend();
    divan::main();
}
