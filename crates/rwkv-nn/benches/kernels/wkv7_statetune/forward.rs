use std::hint::black_box;

use divan::Bencher;
use rwkv_nn::kernels::wkv7_statetune::wkv7_statetune_forward;

#[path = "../../common/mod.rs"]
mod common;

type B = common::BenchBackend;

#[divan::bench(args = common::WKV7_CASES)]
fn bench_wkv7_statetune_forward(bencher: Bencher<'_, '_>, case: &common::Wkv7Case) {
    let device = common::bench_device();
    let input = common::random_wkv7_input::<B>(case, &device);
    let initial_state = common::random_initial_state::<B>(case, &device);

    bencher.bench_local(|| {
        black_box(wkv7_statetune_forward(
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
}

fn main() {
    common::announce_backend();
    divan::main();
}
