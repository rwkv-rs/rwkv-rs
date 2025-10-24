use std::{fmt::Debug, mem::size_of};

use burn::tensor::Element;
use bytemuck::Pod;
use serde::{Deserialize, Serialize};

pub trait TokenUnit: IsDiscrete + Clone + Serialize + Pod + Element {
    const DTYPE: TokenUnitDType;
}

impl TokenUnit for u8 {
    const DTYPE: TokenUnitDType = TokenUnitDType::U8;
}

impl TokenUnit for u16 {
    const DTYPE: TokenUnitDType = TokenUnitDType::U16;
}

impl TokenUnit for f32 {
    const DTYPE: TokenUnitDType = TokenUnitDType::F32;
}

pub trait IsDiscrete {
    const IS_DISCRETE: bool;
}

impl IsDiscrete for u8 {
    const IS_DISCRETE: bool = true;
}

impl IsDiscrete for u16 {
    const IS_DISCRETE: bool = true;
}

impl IsDiscrete for f32 {
    const IS_DISCRETE: bool = false;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TokenUnitDType {
    U8 = 0,
    #[default]
    U16 = 1,
    F32 = 2,
}

impl TokenUnitDType {
    pub fn get_dtype(code: u8) -> Self {
        match code {
            0 => TokenUnitDType::U8,
            1 => TokenUnitDType::U16,
            2 => TokenUnitDType::F32,
            _ => panic!("Unsupported DTYPE code: {}", code),
        }
    }

    pub fn get_token_unit_size(&self) -> usize {
        match self {
            TokenUnitDType::U8 => size_of::<u8>(),
            TokenUnitDType::U16 => size_of::<u16>(),
            TokenUnitDType::F32 => size_of::<f32>(),
        }
    }
}
