use rwkv::custom::backend::Cuda;
use rwkv::custom::tensor::bf16;
use rwkv_lm::model::AutoRegressiveModelConfig;
use rwkv_lm::pth2mpk::{convert_pth_to_mpk, ConvertPthToMpkOptions};

fn main() {
    type MyBackend = Cuda<bf16, i32>;
    let model_path = "../weights/rwkv7-g1a-0.1b-20250728-ctx4096.pth";
    let output_path = "../weights/rwkv7-g1a-0.1b-20250728-ctx4096.mpk";
    let model_config = AutoRegressiveModelConfig::new(12, 65536, 768, 12, 64);
    let option = ConvertPthToMpkOptions::new(model_path, output_path);
    convert_pth_to_mpk::<MyBackend>(&option, model_config);
}
