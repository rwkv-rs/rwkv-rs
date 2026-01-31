use burn::prelude::Backend;
use burn::Tensor;
use burn::tensor::Transaction;
use burn_train::ItemLazy;
use burn_train::metric::{Adaptor, LossInput};
use burn_ndarray::NdArray;

/// Simple classification output adapted for multiple metrics.
#[derive(new)]
pub struct NextTokenPredictionOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for NextTokenPredictionOutput<B> {
    type ItemSync = NextTokenPredictionOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [loss] = Transaction::default()
            .register(self.loss)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        NextTokenPredictionOutput {
            loss: Tensor::from_data(loss, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for NextTokenPredictionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
