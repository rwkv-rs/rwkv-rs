use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use rwkv::config::DatasetFormatOptions;
use rwkv::config::validated::train::FinalTrainConfigBuilder;
use rwkv::custom::data::dataloader::batcher::Batcher;
use rwkv::custom::data::dataloader::{DataLoader, DataLoaderBuilder};
use rwkv::custom::prelude::{Backend, Int, TensorData};
use rwkv::custom::Tensor;
use rwkv::custom::tensor::backend::AutodiffBackend;
use rwkv::data::mmap::dtype::{TokenUnit, TokenUnitDType};
use rwkv::data::mmap::sample::Sampler;
use rwkv::train::data::sliding::{MmapBinReader, SlidingDataset};

#[derive(Clone)]
pub struct AutoRegressiveBatcher<B: Backend, T: TokenUnit> {
    batch_size_per_device: usize,
    context_length: usize,
    num_units_per_token: usize,
    _phantom_backend: PhantomData<B>,
    _phantom_token: PhantomData<T>,
}

impl<B: Backend, T: TokenUnit> AutoRegressiveBatcher<B, T> {
    pub fn new(
        num_units_per_token: usize,
        batch_size_per_device: usize,
        context_length: usize,
    ) -> Self {
        Self {
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


impl<B: Backend, T: TokenUnit> Batcher<B, Vec<T>, AutoRegressiveBatch<B>>
for AutoRegressiveBatcher<B, T>
{
    fn batch(&self, items: Vec<Vec<T>>, device: &B::Device) -> AutoRegressiveBatch<B> {
        let batch_size = items.len();
        let seq_len = items.first().map(|row| row.len()).unwrap_or(0);
        let mut flat = Vec::with_capacity(batch_size.saturating_mul(seq_len));
        for row in items {
            flat.extend(row);
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