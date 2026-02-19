use rwkv::custom::backend::Cuda;
use rwkv::custom::tensor::bf16;
use rwkv_lm::model::AutoRegressiveModelConfig;
use rwkv_lm::pth2mpk::{ConvertPthToMpkOptions, convert_pth_to_mpk};

fn main() {
    type MyBackend = Cuda<bf16, i32>;
    let model_path = "examples/rwkv-lm/weights/rwkv7-g1d-7.2b-20260131-ctx8192.pth";
    let output_path = "examples/rwkv-lm/weights/rwkv7-g1d-7.2b-20260131-ctx8192.mpk";
    let model_config = AutoRegressiveModelConfig::new(32, 65536, 4096, 64, 64);
    let option = ConvertPthToMpkOptions::new(model_path, output_path);
    convert_pth_to_mpk::<MyBackend>(&option, model_config);
}
