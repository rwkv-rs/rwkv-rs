use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::loss::{CrossEntropyLoss, MseLoss, Reduction},
    prelude::Backend,
};
use rwkv_data::mmap::dtype::TokenUnit;
use rwkv_lm::{
    kernels::l2wrap::{L2WrapBackend, l2wrap_apply},
    layers::embedding::TokensOptions,
};

#[derive(Config, Debug)]
pub struct AutoRegressiveLossConfig {
    batch_size_per_device: usize,
    content_length: usize,
    vocabulary_size: usize,
}

impl AutoRegressiveLossConfig {
    pub fn init<B: Backend, T: TokenUnit>(&self, device: &B::Device) -> AutoRegressiveLoss<B> {
        let loss: LossOption<B> = if T::IS_DISCRETE {
            LossOption::CrossEntropyLoss(CrossEntropyLoss::new(None, device))
        } else {
            LossOption::MseLoss(MseLoss::new())
        };
        AutoRegressiveLoss {
            loss,
            flat_len: self.batch_size_per_device * self.content_length,
            vocab_size: self.vocabulary_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct AutoRegressiveLoss<B: Backend> {
    loss: LossOption<B>,

    flat_len: usize,
    vocab_size: usize,
}

impl<B: Backend> AutoRegressiveLoss<B> {
    pub fn forward(
        &self,
        pred: Tensor<B, 3>,
        targets: &TokensOptions<B>,
        apply_l2wrap: bool,
    ) -> Tensor<B, 1>
    where
        B: L2WrapBackend,
    {
        match (&self.loss, targets) {
            (
                LossOption::CrossEntropyLoss(ce_loss),
                TokensOptions::SingleUnitIntTokens(targets),
            ) => {
                let dims = pred.dims();
                debug_assert_eq!(
                    dims[0] * dims[1],
                    self.flat_len,
                    "Prediction tensor batch/context dims must match configured flatten length"
                );
                debug_assert_eq!(
                    dims[2], self.vocab_size,
                    "Prediction last dim must equal vocabulary size"
                );

                let logits = pred.clone().reshape([self.flat_len, self.vocab_size]);
                let targets = targets.clone().reshape([self.flat_len]);

                let loss = ce_loss.forward(logits, targets);

                if apply_l2wrap {
                    l2wrap_apply(loss, pred)
                } else {
                    loss
                }
            },
            (LossOption::MseLoss(mse_loss), TokensOptions::SingleUnitFloatTokens(targets)) => {
                let targets = targets.clone().unsqueeze_dim(2);
                debug_assert_eq!(
                    pred.dims(),
                    targets.dims(),
                    "Prediction and target tensors must be the same shape for MSE loss"
                );
                mse_loss.forward(pred, targets, Reduction::Mean)
            },
            (LossOption::MseLoss(mse_loss), TokensOptions::MultiUnitFloatTokens(targets)) => {
                debug_assert_eq!(
                    pred.dims(),
                    targets.dims(),
                    "Prediction and target tensors must be the same shape for MSE loss"
                );
                mse_loss.forward(pred, targets.clone(), Reduction::Mean)
            },
            _ => panic!("Mismatched loss configuration and target tensor type"),
        }
    }
}

#[derive(Module, Debug)]
enum LossOption<B: Backend> {
    CrossEntropyLoss(CrossEntropyLoss<B>),
    MseLoss(MseLoss),
}
