use burn::{
    backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy},
    tensor::backend::AutodiffBackend,
};
use rwkv_lm::kernels::{l2wrap::L2WrapBackend, wkv7::Wkv7Backend};

mod common;
pub mod custom;
mod ddp;

type RwkvAutodiff<B, C> = Autodiff<B, C>;

pub trait RwkvTrainBackend: AutodiffBackend + Wkv7Backend + L2WrapBackend {}

impl<B, C> RwkvTrainBackend for Autodiff<B, C>
where
    B: Wkv7Backend,
    C: CheckpointStrategy,
{
}
