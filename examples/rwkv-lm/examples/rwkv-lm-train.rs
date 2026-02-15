#![recursion_limit = "256"]

use rwkv::custom::tensor::backend::AutodiffBackend;
use rwkv::nn::kernels::l2wrap::L2WrapBackend;
use rwkv::nn::kernels::wkv7_common::Wkv7Backend;
use rwkv::train::learner::init::{BackendDeviceInit, init_cfg, init_devices, init_log};

#[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
#[allow(unused)]
type ElemType = rwkv::custom::tensor::bf16;
#[cfg(feature = "f32")]
type ElemType = f32;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;

pub fn launch<B: AutodiffBackend + BackendDeviceInit + Wkv7Backend + L2WrapBackend>()
where
    <B as AutodiffBackend>::InnerBackend: BackendDeviceInit + Wkv7Backend + L2WrapBackend,
{
    let (model_cfg_builder, mut train_cfg_builder) = init_cfg(
        "examples/rwkv-lm/config/model.toml",
        "examples/rwkv-lm/config/train.toml",
    );

    let exp_log_path = init_log(&mut train_cfg_builder);

    let devices = init_devices::<B>(&train_cfg_builder);

    rwkv_lm::training::train::<B>(devices, model_cfg_builder, train_cfg_builder, &exp_log_path);
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::{Autodiff, Wgpu};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>();
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use rwkv::custom::backend::{Autodiff, Vulkan};

    pub fn run() {
        launch::<Autodiff<Vulkan<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "metal")]
mod metal {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::{Autodiff, Metal};

    pub fn run() {
        launch::<Autodiff<Metal<ElemType, i32>>>();
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use rwkv::custom::backend::{Autodiff, Cuda};

    pub fn run() {
        launch::<Autodiff<Cuda<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use crate::{ElemType, launch};
    use rwkv::custom::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use rwkv::custom::backend::{Autodiff, Rocm};

    pub fn run() {
        launch::<Autodiff<Rocm<ElemType, i32>, BalancedCheckpointing>>();
    }
}

fn main() {
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "vulkan")]
    vulkan::run();
    #[cfg(feature = "metal")]
    metal::run();
}
