use std::{
    marker::PhantomData,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rwkv_config::validated::train::{FinalTrainConfigBuilder, MmapTokenDtypeOptions, TRAIN_CFG};
use rwkv_data::mmap::{
    bin::BinReader,
    dtype::{TokenUnit, TokenUnitDType},
    sample::Sampler,
};
use rwkv_lm::layers::embedding::TokensOptions;

use crate::data::EPOCH_INDEX;

pub fn get_sliding_data_loaders<B: AutodiffBackend, T: TokenUnit>(
    train_cfg_builder: &mut FinalTrainConfigBuilder,
    devices: &[B::Device],
) -> Vec<Arc<dyn DataLoader<B, AutoRegressiveBatch<B>>>> {
    let bin_path = PathBuf::from(train_cfg_builder.get_dataset_base_path().unwrap())
        .join(train_cfg_builder.get_filename_without_extensions().unwrap())
        .with_extension("bin");

    let bin = Arc::new(BinReader::<T>::new(&bin_path));
    train_cfg_builder.fill_after_read_bin(
        bin.num_tokens as usize,
        bin.num_units_per_token as usize,
        match bin.dtype {
            TokenUnitDType::U8 => MmapTokenDtypeOptions::U8,
            TokenUnitDType::U16 => MmapTokenDtypeOptions::U16,
            TokenUnitDType::F32 => MmapTokenDtypeOptions::F32,
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

pub struct SlidingDataset<T: TokenUnit> {
    pub context_length: u64,

    pub bin: Arc<BinReader<T>>,
    pub sampler: Sampler,
    pub mini_epoch_index: Arc<AtomicUsize>,

    profile_rank0: bool,
}

impl<T: TokenUnit> SlidingDataset<T> {
    pub fn new(
        context_length: u64,

        bin: Arc<BinReader<T>>,
        sampler: Sampler,
        profile_rank0: bool,
    ) -> Self {
        let mini_epoch_index = Arc::new(AtomicUsize::new(0));

        Self {
            bin,
            sampler,
            mini_epoch_index,
            context_length,
            profile_rank0,
        }
    }
}

impl<T: TokenUnit> Dataset<DataItem<T>> for SlidingDataset<T> {
    fn get(&self, index: usize) -> Option<DataItem<T>> {
        let mini_epoch_index = EPOCH_INDEX.load(Ordering::Relaxed);

        let base_offset = self.sampler.get_base_offset(index as u64, mini_epoch_index);

        let token_units = rwkv_bench::hp_block_if!(self.profile_rank0, "data.bin_get", || {
            self.bin
                .get(base_offset * self.context_length, self.context_length + 1)
                .into_owned()
        });

        Some(DataItem { token_units })
    }

    fn len(&self) -> usize {
        TRAIN_CFG.get().unwrap().num_steps_per_mini_epoch_auto
            * TRAIN_CFG.get().unwrap().batch_size_per_device
    }
}

#[derive(Clone, Debug)]

pub struct DataItem<T: TokenUnit> {
    pub token_units: Vec<T>,
}

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
    pub inputs: TokensOptions<B>,
    pub targets: TokensOptions<B>,
}

macro_rules! get_batch_2d {
    ($self:ident, $data:ident, $device:ident, $Scalar:ident, $Variant:ident) => {{
        let tensor =
            rwkv_bench::hp_block_if!($self.profile_rank0, "data.tensor_from_data.2d", || {
                Tensor::<B, 2, $Scalar>::from_data(
                    TensorData::new(
                        $data,
                        [$self.batch_size_per_device, $self.context_length + 1],
                    ),
                    $device,
                )
            });

        let input = tensor
            .clone()
            .slice([0..$self.batch_size_per_device, 0..$self.context_length]);

        let target = tensor.slice([0..$self.batch_size_per_device, 1..$self.context_length + 1]);

        AutoRegressiveBatch {
            inputs: TokensOptions::$Variant(input.clone()),
            targets: TokensOptions::$Variant(target),
        }
    }};
}

macro_rules! get_batch_3d {
    ($self:ident, $data:ident, $device:ident, $Scalar:ident, $Variant:ident) => {{
        let tensor =
            rwkv_bench::hp_block_if!($self.profile_rank0, "data.tensor_from_data.3d", || {
                Tensor::<B, 3, $Scalar>::from_data(
                    TensorData::new(
                        $data,
                        [
                            $self.batch_size_per_device,
                            $self.context_length + 1,
                            $self.num_units_per_token,
                        ],
                    ),
                    $device,
                )
            });

        let input = tensor.clone().slice([
            0..$self.batch_size_per_device,
            0..$self.context_length,
            0..$self.num_units_per_token,
        ]);

        let target = tensor.slice([
            0..$self.batch_size_per_device,
            1..$self.context_length + 1,
            0..$self.num_units_per_token,
        ]);

        AutoRegressiveBatch {
            inputs: TokensOptions::$Variant(input.clone()),
            targets: TokensOptions::$Variant(target),
        }
    }};
}

impl<B: Backend, T: TokenUnit> Batcher<B, DataItem<T>, AutoRegressiveBatch<B>>
    for AutoRegressiveBatcher<B, T>
{
    fn batch(&self, items: Vec<DataItem<T>>, device: &B::Device) -> AutoRegressiveBatch<B> {
        let data: Vec<T> =
            rwkv_bench::hp_block_if!(self.profile_rank0, "data.batch_flatten", || {
                items
                    .iter()
                    .flat_map(|item: &DataItem<T>| item.token_units.iter().copied())
                    .collect()
            });

        match (self.num_units_per_token > 1, T::IS_DISCRETE) {
            (true, true) => {
                get_batch_3d!(self, data, device, Int, MultiUnitIntTokens)
            }
            (true, false) => {
                get_batch_3d!(self, data, device, Float, MultiUnitFloatTokens)
            }
            (false, true) => {
                get_batch_2d!(self, data, device, Int, SingleUnitIntTokens)
            }
            (false, false) => {
                get_batch_2d!(self, data, device, Float, SingleUnitFloatTokens)
            }
        }
    }
}
