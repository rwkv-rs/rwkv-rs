mod grouping;
mod splitter;

use burn::{config::Config, module::AutodiffModule, tensor::backend::AutodiffBackend};
use burn_optim::{
    AdamW, AdamWConfig, GradientsParams, LearningRate, Optimizer, adaptor::OptimizerAdaptor,
    grad_clipping::GradientClippingConfig,
};
use grouping::{ParamGrouperVisitor, ParamGroups};
use splitter::split_grads;

#[derive(Config, Debug)]
pub struct GroupedOptimizerConfig {
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    #[config(default = 5.0)]
    pub high_lr_scale: f64,
    pub grad_clipping: Option<GradientClippingConfig>,
}

impl GroupedOptimizerConfig {
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
        model: &M,
    ) -> GroupedOptimizer<B, M> {
        // Automatically group parameters based on their names and ranks
        let param_groups = ParamGrouperVisitor::group_params(model);

        // Optimizer with weight decay for weight matrices
        let optim_with_wd = AdamWConfig::new()
            .with_beta_1(self.beta_1)
            .with_beta_2(self.beta_2)
            .with_epsilon(self.epsilon)
            .with_weight_decay(self.weight_decay)
            .with_grad_clipping(self.grad_clipping.clone())
            .init();

        // Optimizer without weight decay for biases and other parameters
        let optim_no_wd = AdamWConfig::new()
            .with_beta_1(self.beta_1)
            .with_beta_2(self.beta_2)
            .with_epsilon(self.epsilon)
            .with_weight_decay(0.0) // No weight decay
            .with_grad_clipping(self.grad_clipping.clone())
            .init();

        let high_lr_scale = self.high_lr_scale;

        GroupedOptimizer {
            param_groups,
            optim_with_wd,
            optim_no_wd,
            high_lr_scale,
        }
    }
}

#[derive(Clone)]
pub struct GroupedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    param_groups: ParamGroups,
    optim_with_wd: OptimizerAdaptor<AdamW, M, B>,
    optim_no_wd: OptimizerAdaptor<AdamW, M, B>,
    high_lr_scale: f64,
}

impl<B, M> Optimizer<M, B> for GroupedOptimizer<B, M>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    type Record = (
        <OptimizerAdaptor<AdamW, M, B> as Optimizer<M, B>>::Record,
        <OptimizerAdaptor<AdamW, M, B> as Optimizer<M, B>>::Record,
    );

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        // Split gradients into three groups using a single model visit
        let (high_lr_grads, with_wd_grads, no_wd_grads) =
            split_grads::<B, M>(&module, grads, &self.param_groups);

        // Apply optimizers sequentially
        // high_lr uses no_wd optimizer with higher learning rate
        let module = self
            .optim_no_wd
            .step(lr * self.high_lr_scale, module, high_lr_grads);
        let module = self.optim_with_wd.step(lr, module, with_wd_grads);
        self.optim_no_wd.step(lr, module, no_wd_grads)
    }

    fn to_record(&self) -> Self::Record {
        (self.optim_with_wd.to_record(), self.optim_no_wd.to_record())
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        let (optim_with_wd, optim_no_wd) = record;
        self.optim_with_wd = self.optim_with_wd.load_record(optim_with_wd);
        self.optim_no_wd = self.optim_no_wd.load_record(optim_no_wd);

        self
    }
}
