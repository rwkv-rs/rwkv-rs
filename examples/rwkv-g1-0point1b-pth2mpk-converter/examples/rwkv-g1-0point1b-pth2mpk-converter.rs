use burn::{backend::Cuda, tensor::bf16};
use rwkv_export::pth2mpk;
use rwkv_lm::auto_regressive_model::AutoRegressiveModelConfig;

fn main() {
    type MyBackend = Cuda<bf16, i32>;

    let model_path = "./weights/rwkv7-g1-0.1b-20250307-ctx4096.pth";

    let output_path = "./weights/rwkv7-g1-0.1b-20250307-ctx4096.mpk";

    let model_config = AutoRegressiveModelConfig::new(12, 65536, 768, 12, 64);

    pth2mpk::<MyBackend>(model_path, output_path, model_config);
}
