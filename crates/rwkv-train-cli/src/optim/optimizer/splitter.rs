use std::marker::PhantomData;

use burn::{
    module::{AutodiffModule, ModuleVisitor, Param, ParamId},
    tensor::{Tensor, backend::AutodiffBackend},
};
use burn_optim::GradientsParams;

use super::grouping::ParamGroups;

pub fn split_grads<B, M>(
    model: &M,
    mut source_grads: GradientsParams,
    groups: &ParamGroups,
) -> (GradientsParams, GradientsParams, GradientsParams)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut high_lr_grads = GradientsParams::new();

    let mut with_wd_grads = GradientsParams::new();

    let mut no_wd_grads = GradientsParams::new();

    let mut splitter = GradsSplitter::<B> {
        source: &mut source_grads,
        high_lr: &mut high_lr_grads,
        with_wd: &mut with_wd_grads,
        no_wd: &mut no_wd_grads,
        groups,
        phantom_data: PhantomData,
    };

    model.visit(&mut splitter);

    (high_lr_grads, with_wd_grads, no_wd_grads)
}

struct GradsSplitter<'a, B: AutodiffBackend> {
    source: &'a mut GradientsParams,
    high_lr: &'a mut GradientsParams,
    with_wd: &'a mut GradientsParams,
    no_wd: &'a mut GradientsParams,
    groups: &'a ParamGroups,
    phantom_data: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradsSplitter<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        // Try to remove the gradient from source. If it doesn't exist, skip this
        // parameter.
        if let Some(grad) = self.source.remove::<B::InnerBackend, D>(param.id) {
            // Determine which group this parameter belongs to and register accordingly.
            // Priority: high_lr > with_wd > no_wd (same as rwkv-burn)
            if self.groups.high_lr.contains(&param.id) {
                self.high_lr.register::<B::InnerBackend, D>(param.id, grad);
            } else if self.groups.with_wd.contains(&param.id) {
                self.with_wd.register::<B::InnerBackend, D>(param.id, grad);
            } else {
                self.no_wd.register::<B::InnerBackend, D>(param.id, grad);
            }
        }
    }
}
