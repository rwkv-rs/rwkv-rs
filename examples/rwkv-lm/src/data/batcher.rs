use std::marker::PhantomData;
use std::sync::Arc;
use rwkv::custom::data::dataloader::batcher::Batcher;
use rwkv::custom::prelude::{Backend, Int, TensorData};
use rwkv::custom::Tensor;
use rwkv::data::mmap::dtype::TokenUnit;
use rwkv::train::data::sliding::{MmapBinReader, SlidingSample};

#[derive(Clone)]
pub struct AutoRegressiveBatcher<B: Backend, T: TokenUnit> {
    bin: Arc<MmapBinReader<T>>,
    batch_size_per_device: usize,
    context_length: usize,
    num_units_per_token: usize,
    _phantom_backend: PhantomData<B>,
    _phantom_token: PhantomData<T>,
}

impl<B: Backend, T: TokenUnit> AutoRegressiveBatcher<B, T> {
    pub fn new(
        bin: Arc<MmapBinReader<T>>,
        num_units_per_token: usize,
        batch_size_per_device: usize,
        context_length: usize,
    ) -> Self {
        assert_eq!(num_units_per_token, 1);
        Self {
            bin,
            batch_size_per_device,
            context_length,
            num_units_per_token,

            _phantom_backend: PhantomData,
            _phantom_token: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AutoRegressiveBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}


impl<B: Backend, T: TokenUnit> Batcher<B, SlidingSample, AutoRegressiveBatch<B>>
for AutoRegressiveBatcher<B, T>
{
    fn batch(&self, items: Vec<SlidingSample>, device: &B::Device) -> AutoRegressiveBatch<B> {
        let batch_size = items.len();
        let seq_len = self.context_length + 1;
        let mut flat = Vec::with_capacity(
            batch_size.saturating_mul(seq_len).saturating_mul(self.num_units_per_token),
        );
        for sample in items {
            let token_units = self.bin.get(
                sample.base_offset * self.context_length as u64,
                seq_len as u64,
            );
            flat.extend_from_slice(token_units.as_ref());
        }

        let data = TensorData::new(flat, [batch_size, seq_len]);
        let tensor = Tensor::<B, 2, Int>::from_data(data, device);

        let inputs = tensor
            .clone()
            .slice([0..self.batch_size_per_device, 0..self.context_length]);

        let targets = tensor
            .slice([0..self.batch_size_per_device, 1..self.context_length + 1]);

        AutoRegressiveBatch {
            inputs,
            targets,
        }
    }
}
