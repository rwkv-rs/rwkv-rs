use std::{
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use burn::{data::dataset::Dataset, prelude::*};
use rwkv_config::{DatasetFormatOptions, validated::train::TRAIN_CFG};
use rwkv_data::mmap::{
    bin, bin_old,
    dtype::{TokenUnit, TokenUnitDType},
    sample::Sampler,
};

use crate::data::EPOCH_INDEX;

pub enum MmapBinReader<T: TokenUnit> {
    Rwkv(bin::BinReader<T>),
    RwkvLegacy(bin_old::BinReader<T>),
}

impl<T: TokenUnit> MmapBinReader<T> {
    pub fn open<P: AsRef<std::path::Path>>(path: P, format: DatasetFormatOptions) -> Self {
        match format {
            DatasetFormatOptions::Rwkv => Self::Rwkv(bin::BinReader::new(path)),
            DatasetFormatOptions::RwkvLegacy => Self::RwkvLegacy(bin_old::BinReader::new(path)),
        }
    }

    pub fn num_tokens(&self) -> u64 {
        match self {
            Self::Rwkv(bin) => bin.num_tokens,
            Self::RwkvLegacy(bin) => bin.num_tokens,
        }
    }

    pub fn num_units_per_token(&self) -> u64 {
        match self {
            Self::Rwkv(bin) => bin.num_units_per_token,
            Self::RwkvLegacy(bin) => bin.num_units_per_token,
        }
    }

    pub fn dtype(&self) -> TokenUnitDType {
        match self {
            Self::Rwkv(bin) => bin.dtype,
            Self::RwkvLegacy(bin) => bin.dtype,
        }
    }

    pub fn get_magic_prime(&self, ctx_len: u64) -> u64 {
        match self {
            Self::Rwkv(bin) => bin.get_magic_prime(ctx_len),
            Self::RwkvLegacy(bin) => bin.get_magic_prime(ctx_len),
        }
    }

    pub fn get(&self, offset: u64, length: u64) -> Cow<'_, [T]> {
        match self {
            Self::Rwkv(bin) => bin.get(offset, length),
            Self::RwkvLegacy(bin) => bin.get(offset, length),
        }
    }
}

pub struct SlidingDataset<T: TokenUnit> {
    pub context_length: u64,

    pub bin: Arc<MmapBinReader<T>>,
    pub samplers: Vec<Sampler>,
    pub mini_epoch_index: Arc<AtomicUsize>,

    profile_rank0: bool,
}

impl<T: TokenUnit> SlidingDataset<T> {
    pub fn new(
        context_length: u64,

        bin: Arc<MmapBinReader<T>>,
        samplers: Vec<Sampler>,
        profile_rank0: bool,
    ) -> Self {
        let mini_epoch_index = Arc::new(AtomicUsize::new(0));

        Self {
            bin,
            samplers,
            mini_epoch_index,
            context_length,
            profile_rank0,
        }
    }
}

impl<T: TokenUnit> Dataset<Vec<T>> for SlidingDataset<T> {
    fn get(&self, index: usize) -> Option<Vec<T>> {
        if self.samplers.is_empty() {
            return None;
        }

        let per_device_len = TRAIN_CFG.get().unwrap().num_steps_per_mini_epoch_auto
            * TRAIN_CFG.get().unwrap().batch_size_per_device;
        let num_devices = self.samplers.len();
        let total_len = per_device_len * num_devices;

        if index >= total_len {
            return None;
        }

        let device_index = index / per_device_len;
        let local_index = index % per_device_len;
        let mini_epoch_index = EPOCH_INDEX.load(Ordering::Relaxed);

        let sampler = &self.samplers[device_index];
        let base_offset = sampler.get_base_offset(local_index as u64, mini_epoch_index);

        let profile_rank0 = self.profile_rank0 && device_index == 0;
        let token_units = rwkv_bench::hp_block_if!(profile_rank0, "data.bin_get", || {
            self.bin
                .get(base_offset * self.context_length, self.context_length + 1)
                .into_owned()
        });

        Some(token_units)
    }

    fn len(&self) -> usize {
        if self.samplers.is_empty() {
            return 0;
        }

        let per_device_len = TRAIN_CFG.get().unwrap().num_steps_per_mini_epoch_auto
            * TRAIN_CFG.get().unwrap().batch_size_per_device;
        per_device_len * self.samplers.len()
    }
}
