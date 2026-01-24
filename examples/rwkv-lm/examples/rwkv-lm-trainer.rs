#![recursion_limit = "256"]

#[allow(unused)]
use rwkv::custom::backend::Autodiff;
#[allow(unused)]
use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;

use rwkv::custom::tensor::backend::AutodiffBackend;
use rwkv::train::trainer::common::{init_cfg, init_log};

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

    rwkv_lm::training::train::<B, AgNewsDataset>(
        devices,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
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
    use rwkv::custom::backend::{
        ndarray::{NdArray, NdArrayDevice},
    };

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