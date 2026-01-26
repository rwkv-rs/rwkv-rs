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
    profile_rank0: bool,
    _phantom_backend: PhantomData<B>,
    _phantom_token: PhantomData<T>,
}

impl<B: Backend, T: TokenUnit> AutoRegressiveBatcher<B, T> {
    pub fn new(
        num_units_per_token: usize,
        batch_size_per_device: usize,
        context_length: usize,

        profile_rank0: bool,
    ) -> Self {
        Self {
            batch_size_per_device,
            context_length,
            num_units_per_token,
            profile_rank0,
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


pub fn get_sliding_data_loaders<B: AutodiffBackend, T: TokenUnit>(
    train_cfg_builder: &mut FinalTrainConfigBuilder,
    devices: &[B::Device],
) -> Vec<Arc<dyn DataLoader<B, AutoRegressiveBatch<B>>>> {
    let bin_path = PathBuf::from(train_cfg_builder.get_dataset_base_path().unwrap())
        .join(train_cfg_builder.get_filename_without_extensions().unwrap())
        .with_extension("bin");

    let dataset_format = train_cfg_builder
        .get_dataset_format()
        .unwrap_or(DatasetFormatOptions::Rwkv);
    let bin = Arc::new(MmapBinReader::<T>::open(&bin_path, dataset_format));
    train_cfg_builder.fill_after_read_bin(
        bin.num_tokens() as usize,
        bin.num_units_per_token() as usize,
        match bin.dtype() {
            TokenUnitDType::U8 => TokenUnitDType::U8,
            TokenUnitDType::U16 => TokenUnitDType::U16,
            TokenUnitDType::F32 => TokenUnitDType::F32,
        },
        bin.get_magic_prime(train_cfg_builder.get_context_length().unwrap() as u64) as usize,
    );

    let mut data_loaders = vec![];
    for (device_index, device) in devices.iter().enumerate() {
        let sampler = Sampler::new(
            train_cfg_builder.get_num_devices_per_node().unwrap() as u64,
            device_index as u64,
            (train_cfg_builder
                .get_num_steps_per_mini_epoch_auto()
                .unwrap()
                * train_cfg_builder.get_batch_size_auto().unwrap()) as u64,
            train_cfg_builder.get_magic_prime_auto().unwrap() as u64,
        );

        let profile_rank0 = device_index == 0;

        let dataset = SlidingDataset::new(
            train_cfg_builder.get_context_length().unwrap() as u64,
            bin.clone(),
            sampler,
            profile_rank0,
        );

        let batcher = AutoRegressiveBatcher::<B, T>::new(
            train_cfg_builder.get_mmap_num_units_per_token().unwrap(),
            train_cfg_builder.get_batch_size_per_device().unwrap(),
            train_cfg_builder.get_context_length().unwrap(),
            profile_rank0,
        );

        data_loaders.push(
            DataLoaderBuilder::new(batcher)
                .batch_size(train_cfg_builder.get_batch_size_per_device().unwrap())
                .num_workers(1)
                .set_device(device.clone())
                .build(dataset),
        );
    }

    data_loaders
}
