#![recursion_limit = "256"]

use rwkv::custom::tensor::backend::AutodiffBackend;
use rwkv::nn::kernels::l2wrap::L2WrapBackend;
use rwkv::nn::kernels::wkv7::Wkv7Backend;
use rwkv::train::trainer::common::{BackendDeviceInit, init_cfg, init_devices, init_log};

#[cfg(not(any(feature = "f16", feature = "flex32")))]
#[allow(unused)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;

pub fn launch<B: AutodiffBackend + BackendDeviceInit + Wkv7Backend + L2WrapBackend>()
where
    <B as AutodiffBackend>::InnerBackend: Wkv7Backend + L2WrapBackend,
{
    let (model_cfg_builder, mut train_cfg_builder) = init_cfg(
        "examples/rwkv-lm/config/model.toml",
        "examples/rwkv-lm/config/train.toml",
    );

    let exp_log_path = init_log(&mut train_cfg_builder);

    let devices = init_devices::<B>(&train_cfg_builder);

    rwkv_lm::training::train::<B>(
        devices,
        model_cfg_builder,
        train_cfg_builder,
        &exp_log_path,
    );
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use rwkv::custom::backend::Autodiff;
    use rwkv::custom::backend::ndarray::NdArray;
    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<NdArray<ElemType>>>();
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use rwkv::custom::backend::Autodiff;
    use rwkv::custom::backend::libtorch::LibTorch;
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use crate::{launch, ElemType};

    pub fn run() { launch::<Autodiff<LibTorch<ElemType>>, BalancedCheckpointing>(); }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use rwkv::custom::backend::Autodiff;
    use rwkv::custom::backend::libtorch::LibTorch;
    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<LibTorch<ElemType>>>();
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use rwkv::custom::backend::{Autodiff, Wgpu};
    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>();
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use rwkv::custom::backend::{Autodiff, Vulkan};
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use crate::{launch, ElemType};

    pub fn run() { launch::<Autodiff<Vulkan<ElemType, i32>, BalancedCheckpointing>>(); }
}

#[cfg(feature = "metal")]
mod metal {
    use rwkv::custom::backend::{Autodiff, Metal};
    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<Metal<ElemType, i32>>>();
    }
}

#[cfg(feature = "remote")]
mod remote {
    use rwkv::custom::backend::{Autodiff, RemoteBackend};
    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<RemoteBackend>>();
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use rwkv::custom::backend::{Autodiff, Cuda};
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use crate::{launch, ElemType};

    pub fn run() {
        launch::<Autodiff<Cuda<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use rwkv::custom::backend::{Autodiff, Rocm};
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use crate::{launch, ElemType};

    pub fn run() {launch::<Autodiff<Rocm<ElemType, i32>, BalancedCheckpointing>>(); }
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
