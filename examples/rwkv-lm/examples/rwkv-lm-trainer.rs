#![recursion_limit = "256"]

use rwkv::config::DatasetFormatOptions;
use rwkv::config::validated::train::TokenUnitDType;
#[allow(unused)]
use rwkv::custom::backend::Autodiff;
#[allow(unused)]
use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use std::path::PathBuf;
use std::sync::Arc;

use rwkv::custom::tensor::backend::AutodiffBackend;
use rwkv::data::mmap::dtype::TokenUnitDType;
use rwkv::data::mmap::sample::Sampler;
use rwkv::train::data::sliding::{MmapBinReader, SlidingDataset};
use rwkv::train::trainer::common::{init_cfg, init_devices, init_log};

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;

pub fn launch<B: AutodiffBackend>() {
    let (model_cfg_builder, mut train_cfg_builder) = init_cfg(
        "examples/rwkv-lm/config/model.toml",
        "examples/rwkv-lm/config/train.toml",
    );

    let exp_log_path = init_log(&mut train_cfg_builder);

    let devices = init_devices::<B>(&train_cfg_builder);

    let bin_path = PathBuf::from(train_cfg_builder.get_dataset_base_path().unwrap())
        .join(train_cfg_builder.get_filename_without_extensions().unwrap())
        .with_extension("bin");

    let dataset_format = train_cfg_builder
        .get_dataset_format()
        .unwrap_or(DatasetFormatOptions::Rwkv);
    let bin = Arc::new(MmapBinReader::<u16>::open(&bin_path, dataset_format));
    train_cfg_builder.fill_after_read_bin(
        bin.num_tokens() as usize,
        bin.num_units_per_token() as usize,
        bin.dtype(),
        bin.get_magic_prime(train_cfg_builder.get_context_length().unwrap() as u64) as usize,
    );
    
    let device_index = 0;
    
    let sampler = Sampler::new(
        train_cfg_builder.get_num_devices_per_node().unwrap() as u64,
        device_index as u64,
        (train_cfg_builder
            .get_num_steps_per_mini_epoch_auto()
            .unwrap()
            * train_cfg_builder.get_batch_size_auto().unwrap()) as u64,
        train_cfg_builder.get_magic_prime_auto().unwrap() as u64,
    );

    let dataset = SlidingDataset::new(
        train_cfg_builder.get_context_length().unwrap() as u64,
        bin.clone(),
        sampler,
        profile_rank0,
    );

    rwkv_lm::training::train::<B, AgNewsDataset>(
        devices,
        dataset,
        model_cfg_builder,
        train_cfg_builder,
        "/tmp/text-classification-ag-news",
    );
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use rwkv::custom::backend::ndarray::{NdArray, NdArrayDevice};

    use crate::{ElemType, launch};

    pub fn run() {
        println!("Running NDArray training...");
        launch::<Autodiff<NdArray<ElemType>>>(vec![NdArrayDevice::Cpu]);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        launch::<Autodiff<LibTorch<ElemType>>>(vec![device]);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use rwkv::custom::backend::libtorch::{LibTorch, LibTorchDevice};

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<LibTorch<ElemType>>>(vec![LibTorchDevice::Cpu]);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::Wgpu;

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>(vec![Default::default()]);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Vulkan<ElemType, i32>, BalancedCheckpointing>>(vec![Default::default()]);
    }
}

#[cfg(feature = "metal")]
mod metal {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::{Autodiff, Metal};

    pub fn run() {
        launch::<Autodiff<Metal<ElemType, i32>>>(vec![Default::default()]);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::{Autodiff, RemoteBackend};

    pub fn run() {
        launch::<Autodiff<RemoteBackend>>(vec![Default::default()]);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::{ElemType, launch_multi};
    use rwkv::custom::backend::Cuda;

    pub fn run() {
        launch_multi::<Autodiff<Cuda<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::Rocm;

    pub fn run() {
        launch::<Autodiff<Rocm<ElemType, i32>, BalancedCheckpointing>>(vec![Default::default()]);
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
    #[cfg(feature = "vulkan")]
    vulkan::run();
    #[cfg(feature = "metal")]
    metal::run();
}
